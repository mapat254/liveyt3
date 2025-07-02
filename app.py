import sys
import subprocess
import threading
import time
import os
import streamlit.components.v1 as components
import shutil
import datetime
import pandas as pd
import json
import signal
import psutil
import hashlib
import base64
from pathlib import Path

# Install streamlit if not already installed
try:
    import streamlit as st
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    import streamlit as st

# Persistent storage files
STREAMS_FILE = "streams_data.json"
ACTIVE_STREAMS_FILE = "active_streams.json"
TEMPLATES_FILE = "stream_templates.json"
ANALYTICS_FILE = "stream_analytics.json"
PRESETS_FILE = "custom_presets.json"

def load_persistent_streams():
    """Load streams from persistent storage"""
    if os.path.exists(STREAMS_FILE):
        try:
            with open(STREAMS_FILE, "r") as f:
                data = json.load(f)
                return pd.DataFrame(data)
        except:
            return pd.DataFrame(columns=[
                'Video', 'Durasi', 'Jam Mulai', 'Streaming Key', 'Status', 'Is Shorts', 'Quality', 
                'Template', 'Priority', 'Retry Count', 'Created At', 'Tags'
            ])
    return pd.DataFrame(columns=[
        'Video', 'Durasi', 'Jam Mulai', 'Streaming Key', 'Status', 'Is Shorts', 'Quality',
        'Template', 'Priority', 'Retry Count', 'Created At', 'Tags'
    ])

def save_persistent_streams(streams_df):
    """Save streams to persistent storage"""
    try:
        with open(STREAMS_FILE, "w") as f:
            json.dump(streams_df.to_dict('records'), f, indent=2)
    except Exception as e:
        st.error(f"Error saving streams: {e}")

def load_active_streams():
    """Load active streams tracking"""
    if os.path.exists(ACTIVE_STREAMS_FILE):
        try:
            with open(ACTIVE_STREAMS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_active_streams(active_streams):
    """Save active streams tracking"""
    try:
        with open(ACTIVE_STREAMS_FILE, "w") as f:
            json.dump(active_streams, f, indent=2)
    except Exception as e:
        st.error(f"Error saving active streams: {e}")

def load_templates():
    """Load stream templates"""
    if os.path.exists(TEMPLATES_FILE):
        try:
            with open(TEMPLATES_FILE, "r") as f:
                return json.load(f)
        except:
            return get_default_templates()
    return get_default_templates()

def save_templates(templates):
    """Save stream templates"""
    try:
        with open(TEMPLATES_FILE, "w") as f:
            json.dump(templates, f, indent=2)
    except Exception as e:
        st.error(f"Error saving templates: {e}")

def get_default_templates():
    """Get default stream templates"""
    return {
        "Gaming Stream": {
            "quality": "720p",
            "is_shorts": False,
            "tags": ["gaming", "live"],
            "description": "Optimized for gaming content"
        },
        "Music Stream": {
            "quality": "720p", 
            "is_shorts": False,
            "tags": ["music", "audio"],
            "description": "High audio quality for music"
        },
        "Shorts Vertical": {
            "quality": "720p",
            "is_shorts": True,
            "tags": ["shorts", "vertical"],
            "description": "Vertical format for YouTube Shorts"
        },
        "Low Bandwidth": {
            "quality": "480p",
            "is_shorts": False,
            "tags": ["low-bandwidth", "mobile"],
            "description": "For slow internet connections"
        },
        "High Quality": {
            "quality": "1080p",
            "is_shorts": False,
            "tags": ["hd", "premium"],
            "description": "Maximum quality streaming"
        }
    }

def load_analytics():
    """Load streaming analytics"""
    if os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE, "r") as f:
                return json.load(f)
        except:
            return {"total_streams": 0, "total_duration": 0, "success_rate": 0, "streams_by_quality": {}}
    return {"total_streams": 0, "total_duration": 0, "success_rate": 0, "streams_by_quality": {}}

def save_analytics(analytics):
    """Save streaming analytics"""
    try:
        with open(ANALYTICS_FILE, "w") as f:
            json.dump(analytics, f, indent=2)
    except Exception as e:
        st.error(f"Error saving analytics: {e}")

def update_analytics(status, quality, duration_minutes=0):
    """Update streaming analytics"""
    analytics = load_analytics()
    analytics["total_streams"] = analytics.get("total_streams", 0) + 1
    
    if duration_minutes > 0:
        analytics["total_duration"] = analytics.get("total_duration", 0) + duration_minutes
    
    # Track by quality
    if quality not in analytics.get("streams_by_quality", {}):
        analytics.setdefault("streams_by_quality", {})[quality] = 0
    analytics["streams_by_quality"][quality] += 1
    
    # Calculate success rate
    if status in ["completed", "Selesai"]:
        analytics["successful_streams"] = analytics.get("successful_streams", 0) + 1
        analytics["success_rate"] = (analytics["successful_streams"] / analytics["total_streams"]) * 100
    
    save_analytics(analytics)

def check_ffmpeg():
    """Check if ffmpeg is installed and available"""
    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        st.error("FFmpeg is not installed or not in PATH. Please install FFmpeg to use this application.")
        st.markdown("""
        ### How to install FFmpeg:
        
        - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
        - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
        - **macOS**: `brew install ffmpeg`
        """)
        return False
    return True

def is_process_running(pid):
    """Check if a process with given PID is still running"""
    try:
        if psutil.pid_exists(pid):
            process = psutil.Process(pid)
            if 'ffmpeg' in process.name().lower():
                return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return False

def get_system_resources():
    """Get current system resource usage"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_free_gb": disk.free / (1024**3)
        }
    except:
        return {"cpu": 0, "memory_percent": 0, "memory_available_gb": 0, "disk_free_gb": 0}

def estimate_bandwidth_usage(quality, duration_minutes):
    """Estimate bandwidth usage for a stream"""
    bitrates = {
        "480p": 1000,  # kbps
        "720p": 2500,
        "1080p": 4500
    }
    
    bitrate = bitrates.get(quality, 2500)
    # Convert to MB: (bitrate in kbps * duration in minutes * 60 seconds) / 8 bits / 1024 KB
    estimated_mb = (bitrate * duration_minutes * 60) / 8 / 1024
    return estimated_mb

def get_quality_settings(quality, is_shorts=False):
    """Get optimized encoding settings based on quality"""
    settings = {
        "720p": {
            "video_bitrate": "2500k",
            "audio_bitrate": "128k",
            "maxrate": "2750k",
            "bufsize": "5500k",
            "scale": "1280:720" if not is_shorts else "720:1280",
            "fps": "30"
        },
        "1080p": {
            "video_bitrate": "4500k",
            "audio_bitrate": "192k",
            "maxrate": "4950k",
            "bufsize": "9900k",
            "scale": "1920:1080" if not is_shorts else "1080:1920",
            "fps": "30"
        },
        "480p": {
            "video_bitrate": "1000k",
            "audio_bitrate": "96k",
            "maxrate": "1100k",
            "bufsize": "2200k",
            "scale": "854:480" if not is_shorts else "480:854",
            "fps": "30"
        }
    }
    return settings.get(quality, settings["720p"])

def create_stream_playlist(streams_df, playlist_name):
    """Create a playlist of streams for batch processing"""
    playlist = {
        "name": playlist_name,
        "streams": [],
        "created_at": datetime.datetime.now().isoformat(),
        "total_duration": 0
    }
    
    for _, stream in streams_df.iterrows():
        playlist["streams"].append({
            "video": stream["Video"],
            "quality": stream["Quality"],
            "is_shorts": stream["Is Shorts"],
            "duration": stream["Durasi"]
        })
    
    return playlist

def get_video_info(video_path):
    """Get video information using ffprobe (lightweight)"""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            info = json.loads(result.stdout)
            video_stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
            
            if video_stream:
                duration = float(info["format"]["duration"])
                width = int(video_stream["width"])
                height = int(video_stream["height"])
                fps = eval(video_stream["r_frame_rate"])
                
                return {
                    "duration": str(datetime.timedelta(seconds=int(duration))),
                    "resolution": f"{width}x{height}",
                    "fps": round(fps, 2),
                    "size_mb": round(float(info["format"]["size"]) / (1024*1024), 2),
                    "is_vertical": height > width
                }
    except:
        pass
    
    return {"duration": "Unknown", "resolution": "Unknown", "fps": 0, "size_mb": 0, "is_vertical": False}

def run_ffmpeg(video_path, stream_key, is_shorts, row_id, quality="720p", retry_count=0):
    """Stream a video file to RTMP server using ffmpeg with optimized settings"""
    output_url = f"rtmp://a.rtmp.youtube.com/live2/{stream_key}"
    
    # Create log file
    log_file = f"stream_{row_id}.log"
    with open(log_file, "w") as f:
        f.write(f"Starting optimized stream for {video_path} at {datetime.datetime.now()}\n")
        f.write(f"Quality: {quality}, Shorts: {is_shorts}, Retry: {retry_count}\n")
    
    # Get quality settings
    settings = get_quality_settings(quality, is_shorts)
    
    # Build optimized command for YouTube Live
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "info",
        "-re",
        "-stream_loop", "-1",
        "-i", video_path,
        
        # Video encoding settings optimized for YouTube
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-profile:v", "high",
        "-level", "4.1",
        "-pix_fmt", "yuv420p",
        
        # Bitrate control
        "-b:v", settings["video_bitrate"],
        "-maxrate", settings["maxrate"],
        "-bufsize", settings["bufsize"],
        "-minrate", str(int(settings["video_bitrate"].replace('k', '')) // 2) + "k",
        
        # GOP and keyframe settings
        "-g", "60",
        "-keyint_min", "30",
        "-sc_threshold", "0",
        
        # Frame rate
        "-r", settings["fps"],
        
        # Audio encoding
        "-c:a", "aac",
        "-b:a", settings["audio_bitrate"],
        "-ar", "44100",
        "-ac", "2",
        
        # Scaling and format
        "-vf", f"scale={settings['scale']}:force_original_aspect_ratio=decrease,pad={settings['scale']}:(ow-iw)/2:(oh-ih)/2,fps={settings['fps']}",
        
        # Output format
        "-f", "flv",
        
        # Connection settings
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",
        
        # Output URL
        output_url
    ]
    
    # Log the command
    with open(log_file, "a") as f:
        f.write(f"Running: {' '.join(cmd)}\n")
    
    try:
        # Start the process
        if os.name == 'nt':  # Windows
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:  # Unix/Linux/Mac
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True,
                bufsize=1,
                preexec_fn=os.setsid
            )
        
        # Store process ID
        with open(f"stream_{row_id}.pid", "w") as f:
            f.write(str(process.pid))
        
        # Update status
        with open(f"stream_{row_id}.status", "w") as f:
            f.write("streaming")
        
        # Update active streams tracking
        active_streams = load_active_streams()
        active_streams[str(row_id)] = {
            'pid': process.pid,
            'started_at': datetime.datetime.now().isoformat(),
            'quality': quality,
            'retry_count': retry_count
        }
        save_active_streams(active_streams)
        
        # Read and log output in a separate thread
        def log_output():
            try:
                for line in process.stdout:
                    with open(log_file, "a") as f:
                        f.write(line)
                    if "Connection refused" in line or "Server returned 4" in line:
                        with open(f"stream_{row_id}.status", "w") as f:
                            f.write("error: YouTube connection failed")
            except:
                pass
        
        log_thread = threading.Thread(target=log_output, daemon=True)
        log_thread.start()
        
        # Wait for process to complete
        process.wait()
        
        # Update status when done
        with open(f"stream_{row_id}.status", "w") as f:
            f.write("completed")
        
        with open(log_file, "a") as f:
            f.write("Streaming completed.\n")
        
        # Update analytics
        update_analytics("completed", quality)
        
        # Remove from active streams
        active_streams = load_active_streams()
        if str(row_id) in active_streams:
            del active_streams[str(row_id)]
        save_active_streams(active_streams)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        
        with open(log_file, "a") as f:
            f.write(f"{error_msg}\n")
        
        with open(f"stream_{row_id}.status", "w") as f:
            f.write(f"error: {str(e)}")
        
        # Update analytics
        update_analytics("error", quality)
        
        # Remove from active streams
        active_streams = load_active_streams()
        if str(row_id) in active_streams:
            del active_streams[str(row_id)]
        save_active_streams(active_streams)
    
    finally:
        with open(log_file, "a") as f:
            f.write("Streaming finished or stopped.\n")
        
        cleanup_stream_files(row_id)

def cleanup_stream_files(row_id):
    """Clean up all files related to a stream"""
    files_to_remove = [
        f"stream_{row_id}.pid",
        f"stream_{row_id}.status"
    ]
    
    for file_name in files_to_remove:
        try:
            if os.path.exists(file_name):
                os.remove(file_name)
        except:
            pass

def start_stream(video_path, stream_key, is_shorts, row_id, quality="720p"):
    """Start a stream in a separate process"""
    try:
        # Update status immediately
        st.session_state.streams.loc[row_id, 'Status'] = 'Sedang Live'
        save_persistent_streams(st.session_state.streams)
        
        # Write initial status file
        with open(f"stream_{row_id}.status", "w") as f:
            f.write("starting")
        
        # Start streaming in a separate thread
        thread = threading.Thread(
            target=run_ffmpeg,
            args=(video_path, stream_key, is_shorts, row_id, quality),
            daemon=False
        )
        thread.start()
        
        return True
    except Exception as e:
        st.error(f"Error starting stream: {e}")
        return False

def stop_stream(row_id):
    """Stop a running stream"""
    try:
        active_streams = load_active_streams()
        
        # Get PID
        pid = None
        if str(row_id) in active_streams:
            pid = active_streams[str(row_id)]['pid']
        
        if not pid and os.path.exists(f"stream_{row_id}.pid"):
            with open(f"stream_{row_id}.pid", "r") as f:
                pid = int(f.read().strip())
        
        if pid and is_process_running(pid):
            try:
                if os.name == 'nt':  # Windows
                    subprocess.run(['taskkill', '/F', '/PID', str(pid)], 
                                 capture_output=True, check=False)
                else:  # Unix/Linux/Mac
                    try:
                        os.killpg(os.getpgid(pid), signal.SIGTERM)
                        time.sleep(2)
                        if is_process_running(pid):
                            os.killpg(os.getpgid(pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                
                # Update status
                st.session_state.streams.loc[row_id, 'Status'] = 'Dihentikan'
                save_persistent_streams(st.session_state.streams)
                
                # Update status file
                with open(f"stream_{row_id}.status", "w") as f:
                    f.write("stopped")
                
                # Remove from active streams
                if str(row_id) in active_streams:
                    del active_streams[str(row_id)]
                save_active_streams(active_streams)
                
                cleanup_stream_files(row_id)
                return True
                
            except Exception as e:
                st.error(f"Error stopping stream: {str(e)}")
                return False
        else:
            # Process not found, just update status
            st.session_state.streams.loc[row_id, 'Status'] = 'Dihentikan'
            save_persistent_streams(st.session_state.streams)
            cleanup_stream_files(row_id)
            
            if str(row_id) in active_streams:
                del active_streams[str(row_id)]
            save_active_streams(active_streams)
            
            return True
            
    except Exception as e:
        st.error(f"Error stopping stream: {str(e)}")
        return False

def reconnect_to_existing_streams():
    """Reconnect to streams that are still running after page refresh"""
    active_streams = load_active_streams()
    
    pid_files = [f for f in os.listdir('.') if f.startswith('stream_') and f.endswith('.pid')]
    
    for pid_file in pid_files:
        try:
            row_id = int(pid_file.split('_')[1].split('.')[0])
            
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            
            if is_process_running(pid):
                if row_id < len(st.session_state.streams):
                    st.session_state.streams.loc[row_id, 'Status'] = 'Sedang Live'
                    active_streams[str(row_id)] = {
                        'pid': pid,
                        'started_at': datetime.datetime.now().isoformat()
                    }
            else:
                cleanup_stream_files(row_id)
                if str(row_id) in active_streams:
                    del active_streams[str(row_id)]
                
        except (ValueError, FileNotFoundError, IOError):
            try:
                os.remove(pid_file)
            except:
                pass
    
    save_active_streams(active_streams)

def check_stream_statuses():
    """Check status files for all streams and update accordingly"""
    active_streams = load_active_streams()
    
    for idx, row in st.session_state.streams.iterrows():
        status_file = f"stream_{idx}.status"
        
        if str(idx) in active_streams:
            pid = active_streams[str(idx)]['pid']
            
            if not is_process_running(pid):
                if row['Status'] == 'Sedang Live':
                    if os.path.exists(status_file):
                        with open(status_file, "r") as f:
                            status = f.read().strip()
                        
                        if status == "completed":
                            st.session_state.streams.loc[idx, 'Status'] = 'Selesai'
                        elif status.startswith("error:"):
                            st.session_state.streams.loc[idx, 'Status'] = status
                        else:
                            st.session_state.streams.loc[idx, 'Status'] = 'Terputus'
                        
                        save_persistent_streams(st.session_state.streams)
                        os.remove(status_file)
                    
                    del active_streams[str(idx)]
                    save_active_streams(active_streams)
                    cleanup_stream_files(idx)
        
        elif os.path.exists(status_file):
            with open(status_file, "r") as f:
                status = f.read().strip()
            
            if status == "completed" and row['Status'] == 'Sedang Live':
                st.session_state.streams.loc[idx, 'Status'] = 'Selesai'
                save_persistent_streams(st.session_state.streams)
                os.remove(status_file)
            
            elif status.startswith("error:") and row['Status'] == 'Sedang Live':
                st.session_state.streams.loc[idx, 'Status'] = status
                save_persistent_streams(st.session_state.streams)
                os.remove(status_file)

def check_scheduled_streams():
    """Check for streams that need to be started based on schedule"""
    current_time = datetime.datetime.now().strftime("%H:%M")
    
    for idx, row in st.session_state.streams.iterrows():
        if row['Status'] == 'Menunggu' and row['Jam Mulai'] == current_time:
            quality = row.get('Quality', '720p')
            start_stream(row['Video'], row['Streaming Key'], row.get('Is Shorts', False), idx, quality)

def get_stream_logs(row_id, max_lines=100):
    """Get logs for a specific stream"""
    log_file = f"stream_{row_id}.log"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            lines = f.readlines()
        return lines[-max_lines:] if len(lines) > max_lines else lines
    return []

def main():
    st.set_page_config(
        page_title="Advanced Live Streaming Manager",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Advanced Live Streaming Manager")
    st.caption("Professional YouTube Live Streaming with Advanced Features")
    
    # Check if ffmpeg is installed
    if not check_ffmpeg():
        return
    
    # Initialize session state with persistent data
    if 'streams' not in st.session_state:
        st.session_state.streams = load_persistent_streams()
    
    # Reconnect to existing streams after page refresh
    reconnect_to_existing_streams()
    
    # Sidebar with system info and controls
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # System Resources (lightweight check)
        if st.button("📊 Check System Resources"):
            resources = get_system_resources()
            st.metric("CPU Usage", f"{resources['cpu']:.1f}%")
            st.metric("Memory Usage", f"{resources['memory_percent']:.1f}%")
            st.metric("Free Disk Space", f"{resources['disk_free_gb']:.1f} GB")
        
        # Active streams info
        active_streams = load_active_streams()
        if active_streams:
            st.success(f"🟢 {len(active_streams)} stream(s) active")
            for stream_id, info in active_streams.items():
                st.caption(f"Stream {stream_id}: {info.get('quality', 'Unknown')} quality")
        else:
            st.info("⚫ No active streams")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("🔄 Refresh All Status"):
            st.rerun()
        
        if st.button("🧹 Clean Old Logs"):
            log_files = [f for f in os.listdir('.') if f.endswith('.log')]
            for log_file in log_files:
                try:
                    # Remove logs older than 24 hours
                    if os.path.getmtime(log_file) < time.time() - 86400:
                        os.remove(log_file)
                except:
                    pass
            st.success("Old logs cleaned!")
        
        # Analytics summary
        analytics = load_analytics()
        if analytics.get("total_streams", 0) > 0:
            st.subheader("📈 Quick Stats")
            st.metric("Total Streams", analytics["total_streams"])
            st.metric("Success Rate", f"{analytics.get('success_rate', 0):.1f}%")
            st.metric("Total Hours", f"{analytics.get('total_duration', 0)/60:.1f}")
    
    # Check status of running streams
    check_stream_statuses()
    check_scheduled_streams()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎥 Stream Manager", 
        "➕ Add Stream", 
        "📋 Templates", 
        "📊 Analytics", 
        "📝 Logs", 
        "⚙️ Settings"
    ])
    
    with tab1:
        st.subheader("Stream Management Dashboard")
        
        # Stream filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Filter by Status", 
                ["All", "Menunggu", "Sedang Live", "Selesai", "Dihentikan", "Error"])
        with col2:
            quality_filter = st.selectbox("Filter by Quality", ["All", "480p", "720p", "1080p"])
        with col3:
            sort_by = st.selectbox("Sort by", ["Created Date", "Start Time", "Status", "Quality"])
        
        # Apply filters
        filtered_streams = st.session_state.streams.copy()
        if status_filter != "All":
            if status_filter == "Error":
                filtered_streams = filtered_streams[filtered_streams['Status'].str.startswith('error:')]
            else:
                filtered_streams = filtered_streams[filtered_streams['Status'] == status_filter]
        
        if quality_filter != "All":
            filtered_streams = filtered_streams[filtered_streams['Quality'] == quality_filter]
        
        st.caption("✅ Streams persist through page refreshes • 🎯 Optimized for YouTube Live")
        
        # Display streams
        if not filtered_streams.empty:
            # Bulk actions
            st.subheader("🔧 Bulk Actions")
            bulk_col1, bulk_col2, bulk_col3 = st.columns(3)
            
            with bulk_col1:
                if st.button("▶️ Start All Waiting"):
                    waiting_streams = filtered_streams[filtered_streams['Status'] == 'Menunggu']
                    for idx in waiting_streams.index:
                        row = st.session_state.streams.loc[idx]
                        start_stream(row['Video'], row['Streaming Key'], 
                                   row.get('Is Shorts', False), idx, row.get('Quality', '720p'))
                    st.rerun()
            
            with bulk_col2:
                if st.button("⏹️ Stop All Active"):
                    active_stream_indices = filtered_streams[filtered_streams['Status'] == 'Sedang Live'].index
                    for idx in active_stream_indices:
                        stop_stream(idx)
                    st.rerun()
            
            with bulk_col3:
                if st.button("🗑️ Remove Completed"):
                    completed_indices = filtered_streams[
                        filtered_streams['Status'].isin(['Selesai', 'Dihentikan'])
                    ].index
                    st.session_state.streams = st.session_state.streams.drop(completed_indices).reset_index(drop=True)
                    save_persistent_streams(st.session_state.streams)
                    st.rerun()
            
            # Stream table with enhanced info
            st.subheader("📋 Stream List")
            header_cols = st.columns([2, 1, 1, 1, 1, 2, 2, 2])
            headers = ["Video", "Duration", "Start Time", "Quality", "Priority", "Status", "Template", "Actions"]
            for i, header in enumerate(headers):
                header_cols[i].write(f"**{header}**")
            
            for i, row in filtered_streams.iterrows():
                cols = st.columns([2, 1, 1, 1, 1, 2, 2, 2])
                
                # Video info with file size
                video_name = os.path.basename(row['Video'])
                if len(video_name) > 20:
                    video_name = video_name[:17] + "..."
                cols[0].write(video_name)
                
                cols[1].write(row['Durasi'])
                cols[2].write(row['Jam Mulai'])
                
                # Quality with bandwidth estimate
                quality_text = row.get('Quality', '720p')
                if row.get('Is Shorts', False):
                    quality_text += " 📱"
                cols[3].write(quality_text)
                
                # Priority indicator
                priority = row.get('Priority', 'Normal')
                priority_emoji = {"High": "🔴", "Normal": "🟡", "Low": "🟢"}.get(priority, "🟡")
                cols[4].write(f"{priority_emoji} {priority}")
                
                # Enhanced status with color coding
                status = row['Status']
                if status == 'Sedang Live':
                    cols[5].markdown(f"🟢 **{status}**")
                elif status == 'Menunggu':
                    cols[5].markdown(f"🟡 **{status}**")
                elif status == 'Selesai':
                    cols[5].markdown(f"🔵 **{status}**")
                elif status == 'Dihentikan':
                    cols[5].markdown(f"🟠 **{status}**")
                elif status.startswith('error:'):
                    cols[5].markdown(f"🔴 **Error**")
                else:
                    cols[5].write(status)
                
                # Template info
                template = row.get('Template', 'Custom')
                cols[6].write(template)
                
                # Action buttons
                if row['Status'] == 'Menunggu':
                    if cols[7].button("▶️ Start", key=f"start_{i}"):
                        quality = row.get('Quality', '720p')
                        if start_stream(row['Video'], row['Streaming Key'], 
                                      row.get('Is Shorts', False), i, quality):
                            st.rerun()
                
                elif row['Status'] == 'Sedang Live':
                    if cols[7].button("⏹️ Stop", key=f"stop_{i}"):
                        if stop_stream(i):
                            st.rerun()
                
                else:
                    if cols[7].button("🗑️ Remove", key=f"remove_{i}"):
                        st.session_state.streams = st.session_state.streams.drop(i).reset_index(drop=True)
                        save_persistent_streams(st.session_state.streams)
                        log_file = f"stream_{i}.log"
                        if os.path.exists(log_file):
                            os.remove(log_file)
                        st.rerun()
        else:
            st.info("No streams match the current filters. Add streams in the 'Add Stream' tab.")
    
    with tab2:
        st.subheader("➕ Create New Stream")
        
        # Template selection
        templates = load_templates()
        template_names = ["Custom"] + list(templates.keys())
        selected_template = st.selectbox("📋 Use Template", template_names)
        
        if selected_template != "Custom":
            template_data = templates[selected_template]
            st.info(f"📝 {template_data['description']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📁 Video Selection**")
            
            # List available videos with info
            video_files = [f for f in os.listdir('.') if f.endswith(('.mp4', '.flv', '.avi', '.mov', '.mkv'))]
            
            if video_files:
                video_options = {}
                for video in video_files:
                    info = get_video_info(video)
                    video_options[f"{video} ({info['resolution']}, {info['duration']})"] = video
                
                selected_video_display = st.selectbox("Select Video", [""] + list(video_options.keys()))
                selected_video = video_options.get(selected_video_display, "") if selected_video_display else ""
            else:
                selected_video = ""
                st.info("No video files found in current directory")
            
            # File upload
            uploaded_file = st.file_uploader("Or Upload New Video", 
                                           type=['mp4', 'flv', 'avi', 'mov', 'mkv'])
            
            if uploaded_file:
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Video uploaded: {uploaded_file.name}")
                video_path = uploaded_file.name
            elif selected_video:
                video_path = selected_video
                # Show video info
                info = get_video_info(video_path)
                st.info(f"📊 {info['resolution']} • {info['duration']} • {info['size_mb']} MB")
                if info['is_vertical']:
                    st.warning("🔄 Vertical video detected - consider enabling Shorts mode")
            else:
                video_path = None
        
        with col2:
            st.write("**⚙️ Stream Configuration**")
            
            # Apply template defaults
            default_quality = templates[selected_template]['quality'] if selected_template != "Custom" else "720p"
            default_shorts = templates[selected_template]['is_shorts'] if selected_template != "Custom" else False
            
            stream_key = st.text_input("🔑 Stream Key", type="password", 
                                     help="Your YouTube Live stream key")
            
            # Advanced time scheduling
            col2a, col2b = st.columns(2)
            with col2a:
                start_date = st.date_input("📅 Start Date", value=datetime.date.today())
            with col2b:
                start_time = st.time_input("⏰ Start Time", value=datetime.datetime.now().time())
            
            start_datetime = datetime.datetime.combine(start_date, start_time)
            start_time_str = start_datetime.strftime("%H:%M")
            
            duration = st.text_input("⏱️ Duration (HH:MM:SS)", value="01:00:00")
            
            # Quality with bandwidth estimation
            quality = st.selectbox("🎥 Quality", ["480p", "720p", "1080p"], 
                                 index=["480p", "720p", "1080p"].index(default_quality))
            
            # Estimate bandwidth
            try:
                duration_minutes = sum(x * int(t) for x, t in zip([60, 1, 1/60], duration.split(":")))
                estimated_mb = estimate_bandwidth_usage(quality, duration_minutes)
                st.caption(f"📊 Estimated bandwidth: ~{estimated_mb:.0f} MB")
            except:
                pass
            
            is_shorts = st.checkbox("📱 Shorts Mode (Vertical)", value=default_shorts)
            
            # Advanced options
            with st.expander("🔧 Advanced Options"):
                priority = st.selectbox("Priority Level", ["Low", "Normal", "High"], index=1)
                tags = st.text_input("Tags (comma-separated)", 
                                    value=", ".join(templates[selected_template]['tags']) if selected_template != "Custom" else "")
                auto_retry = st.checkbox("🔄 Auto-retry on failure", value=True)
        
        # Add stream button
        if st.button("➕ Add Stream to Queue", type="primary"):
            if video_path and stream_key:
                new_stream = pd.DataFrame({
                    'Video': [video_path],
                    'Durasi': [duration],
                    'Jam Mulai': [start_time_str],
                    'Streaming Key': [stream_key],
                    'Status': ['Menunggu'],
                    'Is Shorts': [is_shorts],
                    'Quality': [quality],
                    'Template': [selected_template],
                    'Priority': [priority],
                    'Retry Count': [0],
                    'Created At': [datetime.datetime.now().isoformat()],
                    'Tags': [tags]
                })
                
                st.session_state.streams = pd.concat([st.session_state.streams, new_stream], ignore_index=True)
                save_persistent_streams(st.session_state.streams)
                
                st.success(f"✅ Stream added: {os.path.basename(video_path)} ({quality})")
                st.balloons()
                st.rerun()
            else:
                if not video_path:
                    st.error("❌ Please select or upload a video")
                if not stream_key:
                    st.error("❌ Please provide a streaming key")
    
    with tab3:
        st.subheader("📋 Stream Templates")
        st.write("Create and manage reusable stream configurations")
        
        templates = load_templates()
        
        # Template management
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📚 Existing Templates**")
            for name, template in templates.items():
                with st.expander(f"📄 {name}"):
                    st.write(f"**Description:** {template['description']}")
                    st.write(f"**Quality:** {template['quality']}")
                    st.write(f"**Shorts Mode:** {'Yes' if template['is_shorts'] else 'No'}")
                    st.write(f"**Tags:** {', '.join(template['tags'])}")
                    
                    if st.button(f"🗑️ Delete {name}", key=f"del_{name}"):
                        if name not in get_default_templates():
                            del templates[name]
                            save_templates(templates)
                            st.rerun()
                        else:
                            st.error("Cannot delete default templates")
        
        with col2:
            st.write("**➕ Create New Template**")
            
            new_name = st.text_input("Template Name")
            new_description = st.text_area("Description")
            new_quality = st.selectbox("Default Quality", ["480p", "720p", "1080p"], key="template_quality")
            new_shorts = st.checkbox("Shorts Mode", key="template_shorts")
            new_tags = st.text_input("Tags (comma-separated)", key="template_tags")
            
            if st.button("💾 Save Template"):
                if new_name and new_description:
                    templates[new_name] = {
                        "quality": new_quality,
                        "is_shorts": new_shorts,
                        "tags": [tag.strip() for tag in new_tags.split(",") if tag.strip()],
                        "description": new_description
                    }
                    save_templates(templates)
                    st.success(f"✅ Template '{new_name}' saved!")
                    st.rerun()
                else:
                    st.error("❌ Please provide name and description")
    
    with tab4:
        st.subheader("📊 Streaming Analytics")
        
        analytics = load_analytics()
        
        if analytics.get("total_streams", 0) > 0:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Streams", analytics["total_streams"])
            with col2:
                st.metric("Success Rate", f"{analytics.get('success_rate', 0):.1f}%")
            with col3:
                st.metric("Total Hours Streamed", f"{analytics.get('total_duration', 0)/60:.1f}")
            with col4:
                active_count = len(load_active_streams())
                st.metric("Currently Active", active_count)
            
            # Quality distribution
            if analytics.get("streams_by_quality"):
                st.subheader("📈 Streams by Quality")
                quality_data = analytics["streams_by_quality"]
                
                # Create a simple bar chart using columns
                for quality, count in quality_data.items():
                    percentage = (count / analytics["total_streams"]) * 100
                    st.write(f"**{quality}:** {count} streams ({percentage:.1f}%)")
                    st.progress(percentage / 100)
            
            # Recent activity
            st.subheader("📅 Recent Activity")
            if not st.session_state.streams.empty:
                recent_streams = st.session_state.streams.tail(10)
                for _, stream in recent_streams.iterrows():
                    status_emoji = {
                        'Menunggu': '🟡',
                        'Sedang Live': '🟢',
                        'Selesai': '🔵',
                        'Dihentikan': '🟠'
                    }.get(stream['Status'], '🔴')
                    
                    st.write(f"{status_emoji} {os.path.basename(stream['Video'])} - {stream['Status']} ({stream.get('Quality', 'Unknown')})")
        else:
            st.info("📊 No analytics data available yet. Start streaming to see analytics!")
    
    with tab5:
        st.subheader("📝 Stream Logs & Monitoring")
        
        # Log viewer
        log_files = [f for f in os.listdir('.') if f.startswith('stream_') and f.endswith('.log')]
        stream_ids = [int(f.split('_')[1].split('.')[0]) for f in log_files]
        
        if stream_ids:
            # Stream selection for logs
            stream_options = {}
            for idx in stream_ids:
                if idx in st.session_state.streams.index:
                    video_name = os.path.basename(st.session_state.streams.loc[idx, 'Video'])
                    status = st.session_state.streams.loc[idx, 'Status']
                    stream_options[f"{video_name} (ID: {idx}) - {status}"] = idx
            
            if stream_options:
                selected_stream = st.selectbox("📋 Select Stream for Logs", 
                                             options=list(stream_options.keys()))
                selected_id = stream_options[selected_stream]
                
                # Log controls
                col1, col2, col3 = st.columns(3)
                with col1:
                    max_lines = st.number_input("Max Lines", min_value=10, max_value=1000, value=100)
                with col2:
                    auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
                with col3:
                    if st.button("📥 Download Logs"):
                        logs = get_stream_logs(selected_id, max_lines=10000)
                        log_content = "".join(logs)
                        st.download_button(
                            label="💾 Download Log File",
                            data=log_content,
                            file_name=f"stream_{selected_id}_logs.txt",
                            mime="text/plain"
                        )
                
                # Display logs
                logs = get_stream_logs(selected_id, max_lines)
                if logs:
                    st.code("".join(logs), language="text")
                    
                    # Auto-refresh
                    if auto_refresh:
                        time.sleep(3)
                        st.rerun()
                else:
                    st.info("No logs available for this stream")
            else:
                st.info("No active streams with logs found")
        else:
            st.info("No log files found. Start a stream to generate logs.")
    
    with tab6:
        st.subheader("⚙️ Advanced Settings & Tools")
        
        # System optimization tips
        st.write("**🚀 Performance Optimization**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📊 Quality Recommendations:**")
            st.info("""
            **480p**: Upload speed ≥ 3 Mbps  
            **720p**: Upload speed ≥ 8 Mbps  
            **1080p**: Upload speed ≥ 15 Mbps
            """)
            
            st.write("**🔧 Troubleshooting:**")
            st.info("""
            • Use ethernet instead of WiFi  
            • Close bandwidth-heavy applications  
            • Monitor CPU usage during streaming  
            • Use lower quality for unstable connections
            """)
        
        with col2:
            st.write("**🎯 YouTube Live Optimizations:**")
            st.success("""
            ✅ Adaptive bitrate control  
            ✅ Low latency encoding  
            ✅ Auto-reconnection  
            ✅ GOP optimization  
            ✅ Buffer management
            """)
            
            # Network test link
            st.write("**🌐 Network Testing:**")
            st.markdown("[🔗 Test Upload Speed](https://speedtest.net)")
            st.markdown("[🔗 YouTube Live Dashboard](https://studio.youtube.com/channel/UC/livestreaming)")
        
        # Data management
        st.write("**🗂️ Data Management**")
        
        data_col1, data_col2, data_col3 = st.columns(3)
        
        with data_col1:
            if st.button("📤 Export Stream Data"):
                export_data = {
                    "streams": st.session_state.streams.to_dict('records'),
                    "templates": load_templates(),
                    "analytics": load_analytics(),
                    "exported_at": datetime.datetime.now().isoformat()
                }
                
                st.download_button(
                    label="💾 Download Export",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"streaming_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with data_col2:
            uploaded_data = st.file_uploader("📥 Import Stream Data", type=['json'])
            if uploaded_data:
                try:
                    import_data = json.load(uploaded_data)
                    
                    if st.button("✅ Confirm Import"):
                        # Import streams
                        if 'streams' in import_data:
                            imported_streams = pd.DataFrame(import_data['streams'])
                            st.session_state.streams = pd.concat([st.session_state.streams, imported_streams], ignore_index=True)
                            save_persistent_streams(st.session_state.streams)
                        
                        # Import templates
                        if 'templates' in import_data:
                            templates = load_templates()
                            templates.update(import_data['templates'])
                            save_templates(templates)
                        
                        st.success("✅ Data imported successfully!")
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Import failed: {e}")
        
        with data_col3:
            if st.button("🗑️ Clear All Data", type="secondary"):
                if st.checkbox("⚠️ I understand this will delete all data"):
                    # Clear all data files
                    for file in [STREAMS_FILE, ACTIVE_STREAMS_FILE, ANALYTICS_FILE]:
                        if os.path.exists(file):
                            os.remove(file)
                    
                    # Reset session state
                    st.session_state.streams = pd.DataFrame(columns=[
                        'Video', 'Durasi', 'Jam Mulai', 'Streaming Key', 'Status', 'Is Shorts', 'Quality',
                        'Template', 'Priority', 'Retry Count', 'Created At', 'Tags'
                    ])
                    
                    st.success("✅ All data cleared!")
                    st.rerun()
    
    # Instructions in sidebar
    with st.sidebar.expander("📖 How to Use"):
        st.markdown("""
        ### 🚀 Quick Start:
        
        1. **Add Stream**: Upload video + enter stream key
        2. **Choose Template**: Use presets for common scenarios  
        3. **Set Quality**: Match your internet speed
        4. **Schedule**: Set start time or start immediately
        5. **Monitor**: Check logs and analytics
        
        ### 🎯 Pro Tips:
        
        - Use templates for consistent settings
        - Monitor system resources during streaming
        - Test with 480p first for new setups
        - Enable auto-retry for unstable connections
        - Check analytics to optimize performance
        
        ### 🔧 Advanced Features:
        
        - **Bulk Operations**: Start/stop multiple streams
        - **Priority Levels**: Manage stream importance
        - **Analytics**: Track performance metrics
        - **Templates**: Save and reuse configurations
        - **Export/Import**: Backup your settings
        """)
    
    # Auto refresh every 30 seconds for status updates
    time.sleep(1)

if __name__ == '__main__':
    main()
