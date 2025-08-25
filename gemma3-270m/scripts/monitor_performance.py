#!/usr/bin/env python3
"""
Performance monitoring script for Gemma3 270M training on MacBook M4.
Tracks training metrics, hardware utilization, and performance benchmarks.
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import threading
from datetime import datetime, timedelta

import mlx.core as mx
import numpy as np

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️  psutil not available. Install with: pip install psutil")

class PerformanceMonitor:
    """Monitor training performance and hardware utilization."""
    
    def __init__(self, logInterval: int = 5, outputFile: str = None):
        """Initialize the performance monitor."""
        self.logInterval = logInterval
        self.outputFile = outputFile
        self.logger = self.setupLogging()
        self.monitoring = False
        self.metrics = []
        self.startTime = None
        
        # Performance tracking
        self.tokenCount = 0
        self.stepCount = 0
        self.lossHistory = []
        
    def setupLogging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        return logging.getLogger(__name__)
    
    def startMonitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.startTime = datetime.now()
        self.logger.info("🚀 Starting performance monitoring...")
        
        # Start monitoring thread
        monitorThread = threading.Thread(target=self.monitorLoop, daemon=True)
        monitorThread.start()
        
        return monitorThread
    
    def stopMonitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        self.logger.info("⏹️  Performance monitoring stopped")
        
        # Save final metrics
        if self.outputFile:
            self.saveMetrics()
    
    def monitorLoop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                metrics = self.collectMetrics()
                self.metrics.append(metrics)
                
                # Log current metrics
                self.logMetrics(metrics)
                
                # Wait for next interval
                time.sleep(self.logInterval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.logInterval)
    
    def collectMetrics(self) -> Dict[str, Any]:
        """Collect current performance metrics."""
        timestamp = datetime.now()
        
        metrics = {
            "timestamp": timestamp.isoformat(),
            "elapsed_time": (timestamp - self.startTime).total_seconds() if self.startTime else 0
        }
        
        # Hardware metrics
        if PSUTIL_AVAILABLE:
            # CPU metrics
            cpuPercent = psutil.cpu_percent(interval=1)
            metrics["cpu_percent"] = cpuPercent
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics["memory_percent"] = memory.percent
            metrics["memory_used_gb"] = memory.used / (1024**3)
            metrics["memory_available_gb"] = memory.available / (1024**3)
            metrics["memory_total_gb"] = memory.total / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics["disk_percent"] = (disk.used / disk.total) * 100
            metrics["disk_free_gb"] = disk.free / (1024**3)
            
            # Network metrics
            network = psutil.net_io_counters()
            metrics["network_bytes_sent"] = network.bytes_sent
            metrics["network_bytes_recv"] = network.bytes_recv
        
        # MLX metrics
        try:
            deviceInfo = mx.device_info()
            metrics["mlx_device"] = str(deviceInfo)
        except Exception as e:
            metrics["mlx_device"] = f"Error: {e}"
        
        # Training metrics
        metrics["step_count"] = self.stepCount
        metrics["token_count"] = self.tokenCount
        
        if self.lossHistory:
            metrics["current_loss"] = self.lossHistory[-1] if self.lossHistory else None
            metrics["avg_loss"] = np.mean(self.lossHistory[-100:]) if len(self.lossHistory) >= 100 else np.mean(self.lossHistory)
            metrics["min_loss"] = min(self.lossHistory) if self.lossHistory else None
            metrics["max_loss"] = max(self.lossHistory) if self.lossHistory else None
        
        # Performance calculations
        if self.startTime and self.tokenCount > 0:
            elapsedSeconds = (timestamp - self.startTime).total_seconds()
            if elapsedSeconds > 0:
                metrics["tokens_per_second"] = self.tokenCount / elapsedSeconds
                metrics["steps_per_second"] = self.stepCount / elapsedSeconds
        
        return metrics
    
    def logMetrics(self, metrics: Dict[str, Any]):
        """Log current metrics to console."""
        if not metrics:
            return
        
        # Format the log message
        logParts = []
        
        # Training progress
        if "step_count" in metrics:
            logParts.append(f"Step: {metrics['step_count']}")
        
        if "tokens_per_second" in metrics:
            logParts.append(f"Tokens/s: {metrics['tokens_per_second']:.1f}")
        
        # Loss information
        if "current_loss" in metrics and metrics["current_loss"] is not None:
            logParts.append(f"Loss: {metrics['current_loss']:.4f}")
        
        # Hardware utilization
        if "cpu_percent" in metrics:
            logParts.append(f"CPU: {metrics['cpu_percent']:.1f}%")
        
        if "memory_percent" in metrics:
            logParts.append(f"RAM: {metrics['memory_percent']:.1f}%")
        
        if "disk_percent" in metrics:
            logParts.append(f"Disk: {metrics['disk_percent']:.1f}%")
        
        # Log the combined metrics
        if logParts:
            self.logger.info(" | ".join(logParts))
    
    def updateTrainingMetrics(self, stepCount: int = None, tokenCount: int = None, loss: float = None):
        """Update training-specific metrics."""
        if stepCount is not None:
            self.stepCount = stepCount
        
        if tokenCount is not None:
            self.tokenCount = tokenCount
        
        if loss is not None:
            self.lossHistory.append(loss)
    
    def getPerformanceSummary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        if not self.metrics:
            return {}
        
        # Calculate averages and trends
        summary = {
            "monitoring_duration_seconds": self.metrics[-1]["elapsed_time"] if self.metrics else 0,
            "total_metrics_collected": len(self.metrics)
        }
        
        if PSUTIL_AVAILABLE:
            # CPU utilization
            cpuValues = [m.get("cpu_percent", 0) for m in self.metrics if "cpu_percent" in m]
            if cpuValues:
                summary["cpu_avg_percent"] = np.mean(cpuValues)
                summary["cpu_max_percent"] = max(cpuValues)
                summary["cpu_min_percent"] = min(cpuValues)
            
            # Memory utilization
            memoryValues = [m.get("memory_percent", 0) for m in self.metrics if "memory_percent" in m]
            if memoryValues:
                summary["memory_avg_percent"] = np.mean(memoryValues)
                summary["memory_max_percent"] = max(memoryValues)
                summary["memory_min_percent"] = min(memoryValues)
        
        # Training performance
        if self.tokenCount > 0 and self.startTime:
            elapsedSeconds = (datetime.now() - self.startTime).total_seconds()
            if elapsedSeconds > 0:
                summary["overall_tokens_per_second"] = self.tokenCount / elapsedSeconds
                summary["overall_steps_per_second"] = self.stepCount / elapsedSeconds
        
        # Loss statistics
        if self.lossHistory:
            summary["loss_statistics"] = {
                "final_loss": self.lossHistory[-1],
                "avg_loss": np.mean(self.lossHistory),
                "min_loss": min(self.lossHistory),
                "max_loss": max(self.lossHistory),
                "loss_trend": "decreasing" if len(self.lossHistory) > 1 and self.lossHistory[-1] < self.lossHistory[0] else "increasing"
            }
        
        return summary
    
    def saveMetrics(self, outputPath: str = None):
        """Save collected metrics to file."""
        if outputPath is None:
            outputPath = self.outputFile
        
        if not outputPath:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            outputPath = f"./logs/performance_metrics_{timestamp}.json"
        
        outputPath = Path(outputPath)
        outputPath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        saveData = {
            "metadata": {
                "start_time": self.startTime.isoformat() if self.startTime else None,
                "end_time": datetime.now().isoformat(),
                "monitoring_interval_seconds": self.logInterval,
                "total_metrics": len(self.metrics)
            },
            "metrics": self.metrics,
            "summary": self.getPerformanceSummary()
        }
        
        with open(outputPath, 'w') as f:
            json.dump(saveData, f, indent=2, default=str)
        
        self.logger.info(f"💾 Performance metrics saved to {outputPath}")
        return outputPath
    
    def printSummary(self):
        """Print a formatted performance summary."""
        summary = self.getPerformanceSummary()
        
        if not summary:
            print("❌ No performance data available")
            return
        
        print("\n" + "=" * 60)
        print("📊 Performance Monitoring Summary")
        print("=" * 60)
        
        # Monitoring duration
        duration = summary.get("monitoring_duration_seconds", 0)
        if duration > 0:
            durationStr = str(timedelta(seconds=int(duration)))
            print(f"⏱️  Monitoring Duration: {durationStr}")
        
        # Training performance
        if "overall_tokens_per_second" in summary:
            print(f"🚀 Overall Performance: {summary['overall_tokens_per_second']:.1f} tokens/second")
        
        if "overall_steps_per_second" in summary:
            print(f"📈 Training Speed: {summary['overall_steps_per_second']:.3f} steps/second")
        
        # Hardware utilization
        if "cpu_avg_percent" in summary:
            print(f"💻 CPU Utilization: {summary['cpu_avg_percent']:.1f}% (avg)")
        
        if "memory_avg_percent" in summary:
            print(f"🧠 Memory Utilization: {summary['memory_avg_percent']:.1f}% (avg)")
        
        # Loss statistics
        if "loss_statistics" in summary:
            lossStats = summary["loss_statistics"]
            print(f"📉 Loss Statistics:")
            print(f"   Final Loss: {lossStats['final_loss']:.4f}")
            print(f"   Average Loss: {lossStats['avg_loss']:.4f}")
            print(f"   Loss Trend: {lossStats['loss_trend']}")
        
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Monitor performance during Gemma3 270M training")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--output", help="Output file for metrics")
    parser.add_argument("--duration", type=int, help="Monitoring duration in seconds (optional)")
    
    args = parser.parse_args()
    
    print("🚀 Gemma3 270M Performance Monitor")
    print("=" * 50)
    
    # Initialize monitor
    monitor = PerformanceMonitor(
        logInterval=args.interval,
        outputFile=args.output
    )
    
    try:
        # Start monitoring
        monitorThread = monitor.startMonitoring()
        
        # Run for specified duration or until interrupted
        if args.duration:
            print(f"⏰ Monitoring for {args.duration} seconds...")
            time.sleep(args.duration)
        else:
            print("⏰ Monitoring indefinitely. Press Ctrl+C to stop...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⏹️  Interrupted by user")
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
    finally:
        # Stop monitoring and show summary
        monitor.stopMonitoring()
        monitor.printSummary()

if __name__ == "__main__":
    main()
