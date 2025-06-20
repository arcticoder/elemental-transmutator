#!/usr/bin/env python3
"""
Data Logger
===========

Comprehensive data logging system for the photonuclear transmutation experiment.
Handles real-time data collection, storage, analysis, and visualization for
all subsystems during the demonstration.

Key capabilities:
- Multi-threaded data collection from all subsystems
- High-frequency data logging with configurable rates
- Real-time data analysis and alert generation
- Automatic data backup and archival
- Live dashboard and visualization
- Experimental protocol tracking
- Quality control and validation
"""

import json
import time
import logging
import sqlite3
import threading
import queue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import csv
import os

@dataclass
class ExperimentSession:
    """Experiment session metadata."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    operator: str
    pb208_mass_g: float
    target_energy_mev: float
    target_flux: float
    planned_duration_h: float
    protocol_version: str
    notes: str

@dataclass
class DataPoint:
    """Universal data point structure."""
    timestamp: float
    subsystem: str
    parameter: str
    value: float
    unit: str
    quality: str  # GOOD, BAD, UNCERTAIN
    session_id: str

class DataLogger:
    """
    Advanced data logging and analysis system.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the data logger."""
        self.config = config or {"database_path": "experiment_data.db"}
        self.logger = logging.getLogger(__name__)
          # Database setup
        self.db_path = self.config.get("database_path", "experiment_data.db")
        self.backup_dir = Path(self.config.get("backup_directory", "data_backups"))
        self.backup_dir.mkdir(exist_ok=True)
        
        # Data collection
        self.data_queue = queue.Queue()
        self.collection_active = False
        self.collection_threads = []
        
        # Current session
        self.current_session = None
        self.session_start = None
          # Subsystem interfaces
        self.subsystems = {}
        self.sampling_rates = self.config.get("sampling_rates", {})
        
        # Alert system
        self.alert_thresholds = self.config.get("alert_thresholds", {})
        self.active_alerts = set()
        
        # Data analysis
        self.analysis_window = self.config.get("analysis_window_s", 60)
        self.last_analysis = time.time()
        
        # Initialize database
        self.init_database()
        
        self.logger.info("DataLogger initialized")
    
    def init_database(self):
        """Initialize SQLite database schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Experiment sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    operator TEXT,
                    pb208_mass_g REAL,
                    target_energy_mev REAL,
                    target_flux REAL,
                    planned_duration_h REAL,
                    protocol_version TEXT,
                    notes TEXT
                )
            """)
            
            # Raw data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS raw_data (
                    timestamp REAL,
                    subsystem TEXT,
                    parameter TEXT,
                    value REAL,
                    unit TEXT,
                    quality TEXT,
                    session_id TEXT,
                    PRIMARY KEY (timestamp, subsystem, parameter)
                )
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    timestamp REAL PRIMARY KEY,
                    subsystem TEXT,
                    parameter TEXT,
                    alert_type TEXT,
                    value REAL,
                    threshold REAL,
                    message TEXT,
                    session_id TEXT
                )
            """)
            
            # Analysis results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    timestamp REAL PRIMARY KEY,
                    analysis_type TEXT,
                    result_json TEXT,
                    session_id TEXT
                )
            """)
            
            # Create indices for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_data_time ON raw_data(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_raw_data_session ON raw_data(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(timestamp)")
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def register_subsystem(self, name: str, interface: Any):
        """Register a subsystem for data collection."""
        self.subsystems[name] = interface
        self.logger.info(f"Registered subsystem: {name}")
    
    def start_session(self, session_config: Dict) -> str:
        """Start a new experiment session."""
        try:
            # Generate session ID
            session_id = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create session object
            self.current_session = ExperimentSession(
                session_id=session_id,
                start_time=datetime.now(),
                end_time=None,
                operator=session_config.get("operator", "unknown"),
                pb208_mass_g=session_config.get("pb208_mass_g", 0),
                target_energy_mev=session_config.get("target_energy_mev", 16.5),
                target_flux=session_config.get("target_flux", 1e13),
                planned_duration_h=session_config.get("planned_duration_h", 48),
                protocol_version=session_config.get("protocol_version", "1.0"),
                notes=session_config.get("notes", "")
            )
            
            # Store in database
            self._store_session(self.current_session)
            
            # Start data collection
            self.session_start = time.time()
            self.start_data_collection()
            
            self.logger.info(f"Experiment session started: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Session start failed: {e}")
            raise
    
    def end_session(self) -> Dict:
        """End the current experiment session and generate summary."""
        if not self.current_session:
            self.logger.warning("No active session to end")
            return {}
        
        try:
            # Stop data collection
            self.stop_data_collection()
            
            # Update session end time
            self.current_session.end_time = datetime.now()
            self._update_session(self.current_session)
            
            # Generate session summary
            summary = self._generate_session_summary()
            
            # Backup data
            self._backup_session_data()
            
            session_id = self.current_session.session_id
            self.current_session = None
            self.session_start = None
            
            self.logger.info(f"Experiment session ended: {session_id}")
            return summary
            
        except Exception as e:
            self.logger.error(f"Session end failed: {e}")
            return {}
    
    def start_data_collection(self):
        """Start multi-threaded data collection from all subsystems."""
        if self.collection_active:
            self.logger.warning("Data collection already active")
            return
        
        self.collection_active = True
        self.data_queue = queue.Queue()
        
        # Start collection thread for each subsystem
        for subsystem_name, interface in self.subsystems.items():
            sampling_rate = self.sampling_rates.get(subsystem_name, 1.0)  # Default 1 Hz
            
            thread = threading.Thread(
                target=self._collection_worker,
                args=(subsystem_name, interface, sampling_rate),
                name=f"DataCollector_{subsystem_name}"
            )
            thread.start()
            self.collection_threads.append(thread)
        
        # Start data processing thread
        process_thread = threading.Thread(target=self._data_processor)
        process_thread.start()
        self.collection_threads.append(process_thread)
        
        self.logger.info(f"Data collection started for {len(self.subsystems)} subsystems")
    
    def stop_data_collection(self):
        """Stop all data collection threads."""
        if not self.collection_active:
            return
        
        self.collection_active = False
        
        # Wait for all threads to finish
        for thread in self.collection_threads:
            thread.join(timeout=5.0)
        
        self.collection_threads.clear()
        self.logger.info("Data collection stopped")
    
    def log_data_point(self, subsystem: str, parameter: str, value: float, 
                      unit: str, quality: str = "GOOD"):
        """Log a single data point."""
        if not self.current_session:
            return
        
        data_point = DataPoint(
            timestamp=time.time(),
            subsystem=subsystem,
            parameter=parameter,
            value=value,
            unit=unit,
            quality=quality,
            session_id=self.current_session.session_id
        )
        
        self.data_queue.put(data_point)
    
    def get_recent_data(self, subsystem: str, parameter: str, 
                       duration_s: float = 300) -> pd.DataFrame:
        """Get recent data for a specific parameter."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT timestamp, value, quality 
                FROM raw_data 
                WHERE subsystem = ? AND parameter = ? 
                AND timestamp > ? 
                ORDER BY timestamp
            """
            
            cutoff_time = time.time() - duration_s
            df = pd.read_sql_query(query, conn, params=(subsystem, parameter, cutoff_time))
            conn.close()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get recent data: {e}")
            return pd.DataFrame()
    
    def get_session_data(self, session_id: str) -> pd.DataFrame:
        """Get all data for a specific session."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT * FROM raw_data 
                WHERE session_id = ? 
                ORDER BY timestamp, subsystem, parameter
            """
            
            df = pd.read_sql_query(query, conn, params=(session_id,))
            conn.close()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to get session data: {e}")
            return pd.DataFrame()
    
    def generate_live_dashboard_data(self) -> Dict:
        """Generate data for live dashboard display."""
        if not self.current_session:
            return {}
        
        try:
            dashboard_data = {
                "session_info": asdict(self.current_session),
                "runtime_hours": (time.time() - self.session_start) / 3600 if self.session_start else 0,
                "subsystem_status": {},
                "recent_alerts": self._get_recent_alerts(),
                "key_metrics": {}
            }
            
            # Get latest data from each subsystem
            for subsystem in self.subsystems.keys():
                latest_data = self._get_latest_subsystem_data(subsystem)
                dashboard_data["subsystem_status"][subsystem] = latest_data
            
            # Calculate key performance metrics
            dashboard_data["key_metrics"] = self._calculate_key_metrics()
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Dashboard data generation failed: {e}")
            return {}
    
    def create_plots(self, output_dir: str = "plots"):
        """Create standard analysis plots for the current session."""
        if not self.current_session:
            self.logger.warning("No active session for plotting")
            return
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            session_data = self.get_session_data(self.current_session.session_id)
            
            if session_data.empty:
                self.logger.warning("No data available for plotting")
                return
            
            # Convert timestamp to relative time in hours
            start_time = session_data['timestamp'].min()
            session_data['time_h'] = (session_data['timestamp'] - start_time) / 3600
            
            # Beam performance plot
            self._plot_beam_performance(session_data, output_path)
            
            # Target temperature plot
            self._plot_target_temperature(session_data, output_path)
            
            # Conversion progress plot
            self._plot_conversion_progress(session_data, output_path)
            
            # Radiation monitoring plot
            self._plot_radiation_monitoring(session_data, output_path)
            
            self.logger.info(f"Plots created in {output_path}")
            
        except Exception as e:
            self.logger.error(f"Plot creation failed: {e}")
    
    def export_data(self, session_id: str, format: str = "csv", 
                   output_dir: str = "exports") -> str:
        """Export session data in various formats."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            session_data = self.get_session_data(session_id)
            
            if format.lower() == "csv":
                filename = f"{session_id}_data.csv"
                filepath = output_path / filename
                session_data.to_csv(filepath, index=False)
                
            elif format.lower() == "json":
                filename = f"{session_id}_data.json"
                filepath = output_path / filename
                session_data.to_json(filepath, orient="records", date_format="iso")
                
            elif format.lower() == "excel":
                filename = f"{session_id}_data.xlsx"
                filepath = output_path / filename
                
                with pd.ExcelWriter(filepath) as writer:
                    session_data.to_excel(writer, sheet_name="Raw_Data", index=False)
                    
                    # Add summary sheet
                    summary = self._generate_session_summary()
                    summary_df = pd.DataFrame([summary])
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)
            
            self.logger.info(f"Data exported: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Data export failed: {e}")
            return ""
    
    def _collection_worker(self, subsystem_name: str, interface: Any, sampling_rate: float):
        """Worker thread for collecting data from a subsystem."""
        interval = 1.0 / sampling_rate
        
        while self.collection_active:
            try:
                start_time = time.time()
                
                # Get data from subsystem interface
                if hasattr(interface, 'get_status'):
                    status = interface.get_status()
                    timestamp = time.time()
                    
                    # Convert status dict to individual data points
                    for param, value in status.items():
                        if isinstance(value, (int, float)):
                            data_point = DataPoint(
                                timestamp=timestamp,
                                subsystem=subsystem_name,
                                parameter=param,
                                value=float(value),
                                unit="",  # Units would be configured
                                quality="GOOD",
                                session_id=self.current_session.session_id
                            )
                            self.data_queue.put(data_point)
                
                # Maintain sampling rate
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Collection error for {subsystem_name}: {e}")
                time.sleep(1)  # Avoid tight error loop
    
    def _data_processor(self):
        """Process data points from the queue."""
        batch = []
        batch_size = 100
        
        while self.collection_active or not self.data_queue.empty():
            try:
                # Get data point with timeout
                try:
                    data_point = self.data_queue.get(timeout=1.0)
                    batch.append(data_point)
                except queue.Empty:
                    continue
                
                # Process batch when full or periodically
                if len(batch) >= batch_size:
                    self._store_data_batch(batch)
                    self._check_alerts(batch)
                    batch.clear()
                
                # Periodic analysis
                if time.time() - self.last_analysis > self.analysis_window:
                    self._perform_analysis()
                    self.last_analysis = time.time()
                
            except Exception as e:
                self.logger.error(f"Data processing error: {e}")
        
        # Process remaining batch
        if batch:
            self._store_data_batch(batch)
    
    def _store_data_batch(self, batch: List[DataPoint]):
        """Store a batch of data points to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for data_point in batch:
                cursor.execute("""
                    INSERT OR REPLACE INTO raw_data VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    data_point.timestamp,
                    data_point.subsystem,
                    data_point.parameter,
                    data_point.value,
                    data_point.unit,
                    data_point.quality,
                    data_point.session_id
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Data storage failed: {e}")
    
    def _check_alerts(self, batch: List[DataPoint]):
        """Check data points for alert conditions."""
        for data_point in batch:
            key = f"{data_point.subsystem}.{data_point.parameter}"
            
            if key in self.alert_thresholds:
                threshold = self.alert_thresholds[key]
                
                if isinstance(threshold, dict):
                    if "max" in threshold and data_point.value > threshold["max"]:
                        self._trigger_alert(data_point, "HIGH", threshold["max"])
                    elif "min" in threshold and data_point.value < threshold["min"]:
                        self._trigger_alert(data_point, "LOW", threshold["min"])
    
    def _trigger_alert(self, data_point: DataPoint, alert_type: str, threshold: float):
        """Trigger an alert for a data point."""
        alert_key = f"{data_point.subsystem}.{data_point.parameter}.{alert_type}"
        
        # Avoid duplicate alerts
        if alert_key in self.active_alerts:
            return
        
        self.active_alerts.add(alert_key)
        
        message = f"{alert_type} alert: {data_point.subsystem}.{data_point.parameter} = {data_point.value} (threshold: {threshold})"
        
        # Store alert in database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO alerts VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data_point.timestamp,
                data_point.subsystem,
                data_point.parameter,
                alert_type,
                data_point.value,
                threshold,
                message,
                data_point.session_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Alert storage failed: {e}")
        
        self.logger.warning(message)
    
    def _perform_analysis(self):
        """Perform periodic data analysis."""
        try:
            # Real-time conversion efficiency analysis
            conversion_data = self.get_recent_data("target_cell", "conversion_percent", 300)
            
            if not conversion_data.empty:
                latest_conversion = conversion_data['value'].iloc[-1]
                conversion_rate = self._calculate_conversion_rate(conversion_data)
                
                analysis_result = {
                    "analysis_type": "conversion_efficiency",
                    "timestamp": time.time(),
                    "latest_conversion_percent": latest_conversion,
                    "conversion_rate_percent_per_hour": conversion_rate,
                    "projected_final_conversion": self._project_final_conversion(conversion_rate)
                }
                
                self._store_analysis_result(analysis_result)
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
    
    def _calculate_conversion_rate(self, data: pd.DataFrame) -> float:
        """Calculate conversion rate from recent data."""
        if len(data) < 2:
            return 0.0
        
        # Linear regression on recent data
        x = data['timestamp'].values
        y = data['value'].values
        
        # Convert to hours
        x_hours = (x - x[0]) / 3600
        
        if len(x_hours) > 1:
            slope = np.polyfit(x_hours, y, 1)[0]
            return slope
        
        return 0.0
    
    def _project_final_conversion(self, rate: float) -> float:
        """Project final conversion based on current rate."""
        if not self.current_session or not self.session_start:
            return 0.0
        
        elapsed_h = (time.time() - self.session_start) / 3600
        remaining_h = self.current_session.planned_duration_h - elapsed_h
        
        # Get current conversion
        current_data = self.get_recent_data("target_cell", "conversion_percent", 60)
        if current_data.empty:
            return 0.0
        
        current_conversion = current_data['value'].iloc[-1]
        projected_additional = rate * remaining_h
        
        return min(100.0, current_conversion + projected_additional)
    
    def _store_session(self, session: ExperimentSession):
        """Store session metadata in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO experiment_sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.start_time.isoformat(),
                session.end_time.isoformat() if session.end_time else None,
                session.operator,
                session.pb208_mass_g,
                session.target_energy_mev,
                session.target_flux,
                session.planned_duration_h,
                session.protocol_version,
                session.notes
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Session storage failed: {e}")
    
    def _update_session(self, session: ExperimentSession):
        """Update session end time in database."""
        self._store_session(session)  # INSERT OR REPLACE handles updates
    
    def _store_analysis_result(self, result: Dict):
        """Store analysis result in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO analysis_results VALUES (?, ?, ?, ?)
            """, (
                result["timestamp"],
                result["analysis_type"],
                json.dumps(result),
                self.current_session.session_id if self.current_session else ""
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Analysis result storage failed: {e}")
    
    def _generate_session_summary(self) -> Dict:
        """Generate comprehensive session summary."""
        if not self.current_session:
            return {}
        
        try:
            session_data = self.get_session_data(self.current_session.session_id)
            
            summary = {
                "session_id": self.current_session.session_id,
                "duration_hours": (self.current_session.end_time - self.current_session.start_time).total_seconds() / 3600,
                "data_points_collected": len(session_data),
                "subsystems_monitored": session_data['subsystem'].nunique(),
                "parameters_tracked": session_data['parameter'].nunique(),
            }
            
            # Add key performance metrics
            if not session_data.empty:
                conversion_data = session_data[
                    (session_data['subsystem'] == 'target_cell') & 
                    (session_data['parameter'] == 'conversion_percent')
                ]
                
                if not conversion_data.empty:
                    summary["final_conversion_percent"] = conversion_data['value'].iloc[-1]
                    summary["max_conversion_percent"] = conversion_data['value'].max()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Session summary generation failed: {e}")
            return {}
    
    def _backup_session_data(self):
        """Create backup of session data."""
        if not self.current_session:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{self.current_session.session_id}_{timestamp}.csv"
            backup_path = self.backup_dir / backup_filename
            
            session_data = self.get_session_data(self.current_session.session_id)
            session_data.to_csv(backup_path, index=False)
            
            self.logger.info(f"Session data backed up: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Session backup failed: {e}")
    
    def _get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Get recent alerts."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM alerts 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (count,))
            
            alerts = []
            for row in cursor.fetchall():
                alerts.append({
                    "timestamp": row[0],
                    "subsystem": row[1],
                    "parameter": row[2],
                    "alert_type": row[3],
                    "value": row[4],
                    "threshold": row[5],
                    "message": row[6]
                })
            
            conn.close()
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to get recent alerts: {e}")
            return []
    
    def _get_latest_subsystem_data(self, subsystem: str) -> Dict:
        """Get latest data point for each parameter in a subsystem."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT parameter, value, timestamp, quality
                FROM raw_data 
                WHERE subsystem = ? AND session_id = ?
                AND timestamp = (
                    SELECT MAX(timestamp) 
                    FROM raw_data rd2 
                    WHERE rd2.subsystem = raw_data.subsystem 
                    AND rd2.parameter = raw_data.parameter
                    AND rd2.session_id = raw_data.session_id
                )
            """, (subsystem, self.current_session.session_id if self.current_session else ""))
            
            data = {}
            for row in cursor.fetchall():
                data[row[0]] = {
                    "value": row[1],
                    "timestamp": row[2],
                    "quality": row[3]
                }
            
            conn.close()
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to get latest subsystem data: {e}")
            return {}
    
    def _calculate_key_metrics(self) -> Dict:
        """Calculate key performance metrics."""
        metrics = {}
        
        try:
            # Conversion efficiency
            conversion_data = self.get_recent_data("target_cell", "conversion_percent", 60)
            if not conversion_data.empty:
                metrics["current_conversion_percent"] = conversion_data['value'].iloc[-1]
            
            # Beam stability
            stability_data = self.get_recent_data("gamma_beam", "stability_percent", 300)
            if not stability_data.empty:
                metrics["beam_stability_percent"] = stability_data['value'].mean()
            
            # Target temperature
            temp_data = self.get_recent_data("target_cell", "target_temperature_c", 60)
            if not temp_data.empty:
                metrics["target_temperature_c"] = temp_data['value'].iloc[-1]
            
            # Uptime
            if self.session_start:
                metrics["runtime_hours"] = (time.time() - self.session_start) / 3600
            
        except Exception as e:
            self.logger.error(f"Key metrics calculation failed: {e}")
        
        return metrics
    
    def _plot_beam_performance(self, data: pd.DataFrame, output_path: Path):
        """Create beam performance plots."""
        beam_data = data[data['subsystem'] == 'gamma_beam']
        
        if beam_data.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Gamma Beam Performance')
        
        # Energy stability
        energy_data = beam_data[beam_data['parameter'] == 'beam_energy_MeV']
        if not energy_data.empty:
            axes[0,0].plot(energy_data['time_h'], energy_data['value'])
            axes[0,0].set_title('Beam Energy')
            axes[0,0].set_ylabel('Energy (MeV)')
        
        # Flux stability
        flux_data = beam_data[beam_data['parameter'] == 'beam_flux']
        if not flux_data.empty:
            axes[0,1].plot(flux_data['time_h'], flux_data['value'])
            axes[0,1].set_title('Beam Flux')
            axes[0,1].set_ylabel('Flux (γ/cm²/s)')
            axes[0,1].set_yscale('log')
        
        # Beam stability
        stability_data = beam_data[beam_data['parameter'] == 'stability_percent']
        if not stability_data.empty:
            axes[1,0].plot(stability_data['time_h'], stability_data['value'])
            axes[1,0].set_title('Beam Stability')
            axes[1,0].set_ylabel('Stability (%)')
        
        # Dose rate
        dose_data = beam_data[beam_data['parameter'] == 'dose_rate_usv_h']
        if not dose_data.empty:
            axes[1,1].plot(dose_data['time_h'], dose_data['value'])
            axes[1,1].set_title('Dose Rate')
            axes[1,1].set_ylabel('Dose Rate (μSv/h)')
        
        for ax in axes.flat:
            ax.set_xlabel('Time (hours)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'beam_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_target_temperature(self, data: pd.DataFrame, output_path: Path):
        """Create target temperature plot."""
        target_data = data[data['subsystem'] == 'target_cell']
        
        temp_data = target_data[target_data['parameter'] == 'target_temperature_c']
        coolant_in = target_data[target_data['parameter'] == 'coolant_temp_in_c']
        coolant_out = target_data[target_data['parameter'] == 'coolant_temp_out_c']
        
        plt.figure(figsize=(10, 6))
        
        if not temp_data.empty:
            plt.plot(temp_data['time_h'], temp_data['value'], label='Target Temperature', linewidth=2)
        if not coolant_in.empty:
            plt.plot(coolant_in['time_h'], coolant_in['value'], label='Coolant In', alpha=0.7)
        if not coolant_out.empty:
            plt.plot(coolant_out['time_h'], coolant_out['value'], label='Coolant Out', alpha=0.7)
        
        plt.title('Target Cell Temperature Profile')
        plt.xlabel('Time (hours)')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'target_temperature.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_conversion_progress(self, data: pd.DataFrame, output_path: Path):
        """Create conversion progress plot."""
        target_data = data[data['subsystem'] == 'target_cell']
        
        conversion_data = target_data[target_data['parameter'] == 'conversion_percent']
        au_activity = target_data[target_data['parameter'] == 'au197_activity_bq']
        
        if conversion_data.empty:
            return
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Conversion percentage
        ax1.plot(conversion_data['time_h'], conversion_data['value'], 'b-', linewidth=2, label='Conversion %')
        ax1.set_xlabel('Time (hours)')
        ax1.set_ylabel('Conversion (%)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Au-197 activity on secondary axis
        if not au_activity.empty:
            ax2 = ax1.twinx()
            ax2.plot(au_activity['time_h'], au_activity['value'], 'r-', alpha=0.7, label='Au-197 Activity')
            ax2.set_ylabel('Au-197 Activity (Bq)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_yscale('log')
        
        plt.title('Transmutation Progress')
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'conversion_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_radiation_monitoring(self, data: pd.DataFrame, output_path: Path):
        """Create radiation monitoring plots."""
        target_data = data[data['subsystem'] == 'target_cell']
        
        neutron_flux = target_data[target_data['parameter'] == 'neutron_flux']
        gamma_dose = target_data[target_data['parameter'] == 'gamma_dose_rate']
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Neutron flux
        if not neutron_flux.empty:
            axes[0].plot(neutron_flux['time_h'], neutron_flux['value'])
            axes[0].set_title('Neutron Flux')
            axes[0].set_ylabel('Flux (n/cm²/s)')
            axes[0].set_yscale('log')
            axes[0].grid(True, alpha=0.3)
        
        # Gamma dose rate
        if not gamma_dose.empty:
            axes[1].plot(gamma_dose['time_h'], gamma_dose['value'])
            axes[1].set_title('Gamma Dose Rate')
            axes[1].set_xlabel('Time (hours)')
            axes[1].set_ylabel('Dose Rate (Gy/h)')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'radiation_monitoring.png', dpi=300, bbox_inches='tight')
        plt.close()

    def cleanup(self) -> None:
        """Cleanup method for integration tests."""
        # Stop data collection
        self.collection_active = False
        # Wait for threads to finish
        for thread in self.collection_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        # Close database connection
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    config = {
        "database_path": "test_experiment.db",
        "backup_directory": "test_backups",
        "sampling_rates": {
            "gamma_beam": 1.0,
            "target_cell": 0.5,
            "safety_system": 2.0
        },
        "alert_thresholds": {
            "target_cell.target_temperature_c": {"max": 300},
            "gamma_beam.stability_percent": {"min": 90}
        }
    }
    
    # Initialize data logger
    logger = DataLogger(config)
    
    try:
        # Start test session
        session_config = {
            "operator": "test_operator",
            "pb208_mass_g": 10.0,
            "target_energy_mev": 16.5,
            "planned_duration_h": 1.0,
            "notes": "Test session"
        }
        
        session_id = logger.start_session(session_config)
        print(f"✓ Session started: {session_id}")
        
        # Simulate data collection for 30 seconds
        print("Collecting data for 30 seconds...")
        for i in range(30):
            # Simulate gamma beam data
            logger.log_data_point("gamma_beam", "beam_energy_MeV", 16.5 + np.random.normal(0, 0.1), "MeV")
            logger.log_data_point("gamma_beam", "stability_percent", 98 + np.random.normal(0, 1), "%")
            
            # Simulate target cell data
            logger.log_data_point("target_cell", "target_temperature_c", 200 + np.random.normal(0, 5), "°C")
            logger.log_data_point("target_cell", "conversion_percent", i * 0.1, "%")
            
            time.sleep(1)
        
        # Generate dashboard data
        dashboard = logger.generate_live_dashboard_data()
        print(f"✓ Dashboard data: {len(dashboard)} sections")
        
        # End session and get summary
        summary = logger.end_session()
        print(f"✓ Session ended, summary: {summary}")
        
        # Export data
        export_file = logger.export_data(session_id, "csv")
        print(f"✓ Data exported: {export_file}")
        
    except KeyboardInterrupt:
        print("\nStopping data logger...")
        logger.end_session()
