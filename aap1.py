import streamlit as st
import speech_recognition as sr
import requests
import json
import threading
import queue
import time
import numpy as np
from datetime import datetime
import atexit
import io
import os
import logging
import base64

# Set up logging to suppress warnings
logging.getLogger("speechrecognition").setLevel(logging.ERROR)

# Global variables for thread-safe communication
debug_queue = queue.Queue()
conversation_queue = queue.Queue()
audio_queue = queue.Queue()

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")  # For TTS
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"

def initialize_session_state():
    """Initialize all session state variables"""
    if 'debug_info' not in st.session_state:
        st.session_state.debug_info = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'is_listening' not in st.session_state:
        st.session_state.is_listening = False
    if 'is_continuous_listening' not in st.session_state:
        st.session_state.is_continuous_listening = False
    if 'audio_processor' not in st.session_state:
        st.session_state.audio_processor = None
    if 'microphone_working' not in st.session_state:
        st.session_state.microphone_working = False
    if 'tts_enabled' not in st.session_state:
        st.session_state.tts_enabled = False
    if 'selected_voice' not in st.session_state:
        st.session_state.selected_voice = "alloy"
    if 'current_listening_status' not in st.session_state:
        st.session_state.current_listening_status = "Idle"
    if 'last_transcription' not in st.session_state:
        st.session_state.last_transcription = ""
    if 'last_ai_response' not in st.session_state:
        st.session_state.last_ai_response = ""

def add_debug_info_threadsafe(message):
    """Thread-safe way to add debug info"""
    try:
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        debug_msg = f"{timestamp} {message}"
        debug_queue.put(debug_msg)
    except Exception as e:
        print(f"Debug queue error: {e}")

def add_conversation_threadsafe(user_text, ai_response):
    """Thread-safe way to add conversation"""
    try:
        conversation_queue.put({"user": user_text, "ai": ai_response, "timestamp": datetime.now()})
    except Exception as e:
        print(f"Conversation queue error: {e}")

def update_debug_info():
    """Update debug info from queue - call this from main thread only"""
    updated = False
    while not debug_queue.empty():
        try:
            debug_msg = debug_queue.get_nowait()
            st.session_state.debug_info.append(debug_msg)
            updated = True
            # Keep only last 100 debug messages to prevent memory issues
            if len(st.session_state.debug_info) > 100:
                st.session_state.debug_info = st.session_state.debug_info[-100:]
        except queue.Empty:
            break
        except Exception as e:
            print(f"Error updating debug info: {e}")
            break
    return updated

def update_conversation_history():
    """Update conversation history from queue - call this from main thread only"""
    updated = False
    while not conversation_queue.empty():
        try:
            conv_item = conversation_queue.get_nowait()
            st.session_state.conversation_history.append(conv_item)
            updated = True
            # Keep only last 50 conversations
            if len(st.session_state.conversation_history) > 50:
                st.session_state.conversation_history = st.session_state.conversation_history[-50:]
        except queue.Empty:
            break
        except Exception as e:
            print(f"Error updating conversation history: {e}")
            break
    return updated

def test_microphone():
    """Test microphone once - not continuously"""
    try:
        r = sr.Recognizer()
        r.energy_threshold = 300  # Set a reasonable threshold
        r.dynamic_energy_threshold = True
        
        with sr.Microphone() as source:
            add_debug_info_threadsafe("Testing microphone...")
            r.adjust_for_ambient_noise(source, duration=1.0)
            add_debug_info_threadsafe("Microphone calibrated for ambient noise")
        
        add_debug_info_threadsafe("Microphone test successful")
        return True
    except Exception as e:
        add_debug_info_threadsafe(f"Microphone test failed: {str(e)}")
        return False

def get_ai_response(text):
    """Get response from Groq API"""
    try:
        add_debug_info_threadsafe(f"Sending request to Groq API: {text[:50]}...")
        
        if GROQ_API_KEY == "your_groq_api_key_here":
            add_debug_info_threadsafe("API key not configured")
            return "Please configure your Groq API key in the environment variables."
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama3-8b-8192",
            "messages": [
                {"role": "system", "content": "You are a helpful voice assistant. Provide concise, friendly responses."},
                {"role": "user", "content": text}
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            ai_response = result['choices'][0]['message']['content'].strip()
            add_debug_info_threadsafe(f"Got response from Groq: {ai_response[:50]}...")
            return ai_response
        else:
            error_msg = f"API Error: {response.status_code}"
            if response.text:
                error_msg += f" - {response.text[:100]}"
            add_debug_info_threadsafe(error_msg)
            return "Sorry, I encountered an error while processing your request."
            
    except requests.exceptions.Timeout:
        add_debug_info_threadsafe("Request timeout")
        return "Sorry, the request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        add_debug_info_threadsafe("Connection error")
        return "Sorry, I'm having trouble connecting to the AI service."
    except Exception as e:
        add_debug_info_threadsafe(f"Error getting AI response: {str(e)}")
        return "Sorry, I encountered an unexpected error."

def generate_tts_audio_openai(text, voice="alloy"):
    """Generate TTS audio using OpenAI API"""
    try:
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
            add_debug_info_threadsafe("OpenAI API key not configured for TTS")
            return None
            
        add_debug_info_threadsafe(f"Generating TTS for: {text[:50]}...")
        
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": "mp3"
        }
        
        response = requests.post(OPENAI_TTS_URL, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            add_debug_info_threadsafe("TTS audio generated successfully")
            return response.content
        else:
            add_debug_info_threadsafe(f"TTS API Error: {response.status_code}")
            return None
            
    except Exception as e:
        add_debug_info_threadsafe(f"TTS generation failed: {str(e)}")
        return None

def generate_tts_fallback(text):
    """Fallback TTS using browser's speech synthesis (for demo purposes)"""
    try:
        # Create a simple HTML audio element with speech synthesis
        # This is a workaround when OpenAI API is not available
        add_debug_info_threadsafe("Using fallback TTS (browser speech synthesis)")
        
        # Return HTML/JS code that will trigger browser TTS
        escaped_text = text.replace('"', '\\"')
        html_code = f"""
        <script>
        if ('speechSynthesis' in window) {{
            const utterance = new SpeechSynthesisUtterance("{escaped_text}");
            utterance.rate = 0.8;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            speechSynthesis.speak(utterance);
        }}
        </script>
        """
        return html_code
        
    except Exception as e:
        add_debug_info_threadsafe(f"Fallback TTS failed: {str(e)}")
        return None

def listen_for_speech():
    """Listen for speech input with improved error handling"""
    try:
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.8
        r.phrase_threshold = 0.3
        
        with sr.Microphone() as source:
            add_debug_info_threadsafe("Listening for speech...")
            # Update status for continuous listening
            if hasattr(st.session_state, 'current_listening_status'):
                st.session_state.current_listening_status = "Listening..."
            
            # Shorter ambient noise adjustment for continuous listening
            r.adjust_for_ambient_noise(source, duration=0.5)
            
            # Listen with timeout
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
            
        add_debug_info_threadsafe("Processing speech...")
        st.session_state.current_listening_status = "Processing..."
        
        # Try multiple recognition services
        text = None
        try:
            text = r.recognize_google(audio)
            add_debug_info_threadsafe(f"Google recognized: {text}")
            st.session_state.last_transcription = text
        except sr.UnknownValueError:
            add_debug_info_threadsafe("Google could not understand audio")
        except sr.RequestError as e:
            add_debug_info_threadsafe(f"Google recognition error: {e}")
        
        return text
        
    except sr.WaitTimeoutError:
        add_debug_info_threadsafe("Listening timeout - no speech detected")
        st.session_state.current_listening_status = "Waiting for speech..."
        return None
    except sr.UnknownValueError:
        add_debug_info_threadsafe("Could not understand audio")
        return None
    except Exception as e:
        add_debug_info_threadsafe(f"Speech recognition error: {str(e)}")
        return None

class AudioProcessor:
    def __init__(self):
        self.is_running = False
        self.audio_thread = None
        
    def start_continuous_listening(self):
        """Start continuous listening"""
        if self.is_running:
            add_debug_info_threadsafe("Continuous listening already running")
            return
        
        self.is_running = True
        self.audio_thread = threading.Thread(target=self._audio_processing_loop, daemon=True)
        self.audio_thread.start()
        add_debug_info_threadsafe("Continuous listening started successfully")
    
    def stop_continuous_listening(self):
        """Stop continuous listening"""
        if not self.is_running:
            return
            
        add_debug_info_threadsafe("Stopping continuous listening")
        self.is_running = False
        st.session_state.current_listening_status = "Stopping..."
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=3.0)
            
        st.session_state.current_listening_status = "Idle"
        add_debug_info_threadsafe("Continuous listening stopped")
    
    def _audio_processing_loop(self):
        """Main audio processing loop with better error handling"""
        add_debug_info_threadsafe("Audio processing thread started")
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.is_running:
            try:
                # Listen for speech
                text = listen_for_speech()
                
                if text and self.is_running:
                    add_debug_info_threadsafe(f"Handling conversation: {text}")
                    consecutive_errors = 0  # Reset error counter on success
                    
                    st.session_state.current_listening_status = "Getting AI response..."
                    
                    # Get AI response
                    ai_response = get_ai_response(text)
                    
                    if ai_response and self.is_running:
                        st.session_state.last_ai_response = ai_response
                        
                        # Add to conversation history
                        add_conversation_threadsafe(text, ai_response)
                        
                        # Generate TTS if enabled
                        if st.session_state.get('tts_enabled', False):
                            st.session_state.current_listening_status = "Generating speech..."
                            
                            # Try OpenAI TTS first, then fallback
                            audio_data = generate_tts_audio_openai(ai_response, st.session_state.get('selected_voice', 'alloy'))
                            
                            if audio_data:
                                audio_queue.put(("audio", audio_data))
                            else:
                                # Use fallback TTS
                                fallback_html = generate_tts_fallback(ai_response)
                                if fallback_html:
                                    audio_queue.put(("html", fallback_html))
                        
                        st.session_state.current_listening_status = "Ready for next input..."
                
                # Small delay to prevent excessive CPU usage
                time.sleep(1.0)
                
            except Exception as e:
                consecutive_errors += 1
                add_debug_info_threadsafe(f"Error in audio processing: {str(e)}")
                
                if consecutive_errors >= max_consecutive_errors:
                    add_debug_info_threadsafe("Too many consecutive errors, stopping continuous listening")
                    self.is_running = False
                    st.session_state.current_listening_status = "Error - Stopped"
                    break
                
                time.sleep(2)  # Wait longer before retrying on error
        
        add_debug_info_threadsafe("Audio processing thread ended")

def handle_single_conversation():
    """Handle a single conversation interaction"""
    if not st.session_state.microphone_working:
        st.error("Microphone is not working. Please test your microphone first.")
        return
    
    # Create placeholder for status updates
    status_placeholder = st.empty()
    result_placeholder = st.empty()
    
    try:
        status_placeholder.info("🎤 Listening... Speak now!")
        text = listen_for_speech()
        
        if text:
            status_placeholder.success(f"✅ You said: {text}")
            
            with st.spinner("🤖 Getting AI response..."):
                ai_response = get_ai_response(text)
            
            if ai_response:
                result_placeholder.info(f"🤖 AI: {ai_response}")
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    "user": text,
                    "ai": ai_response,
                    "timestamp": datetime.now()
                })
                
                # Generate TTS if enabled
                if st.session_state.get('tts_enabled', False):
                    with st.spinner("🔊 Generating speech..."):
                        # Try OpenAI TTS first
                        audio_data = generate_tts_audio_openai(ai_response, st.session_state.get('selected_voice', 'alloy'))
                        
                        if audio_data:
                            st.audio(audio_data, format='audio/mp3')
                        else:
                            # Use fallback browser TTS
                            st.warning("OpenAI TTS not available, using browser speech synthesis...")
                            fallback_html = generate_tts_fallback(ai_response)
                            if fallback_html:
                                st.components.v1.html(fallback_html, height=0)
        else:
            status_placeholder.warning("⚠️ No speech detected or could not understand. Please try again.")
            
    except Exception as e:
        st.error(f"Error during conversation: {str(e)}")
        add_debug_info_threadsafe(f"Single conversation error: {str(e)}")

def main():
    st.set_page_config(
        page_title="Voice Assistant",
        page_icon="🎤",
        layout="wide"
    )
    
    st.title("🎤 Advanced Voice Assistant")
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize audio processor
    if st.session_state.audio_processor is None:
        st.session_state.audio_processor = AudioProcessor()
    
    # Update queues and check for updates
    debug_updated = update_debug_info()
    conv_updated = update_conversation_history()
    
    # Create main layout
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.subheader("🎛️ Controls")
        
        # API Configuration
        with st.expander("⚙️ Configuration", expanded=False):
            # Groq API Key
            groq_key_input = st.text_input("Groq API Key", type="password", value=GROQ_API_KEY if GROQ_API_KEY != "your_groq_api_key_here" else "")
            if groq_key_input:
                os.environ["GROQ_API_KEY"] = groq_key_input
                globals()["GROQ_API_KEY"] = groq_key_input
            
            # OpenAI API Key for TTS
            openai_key_input = st.text_input("OpenAI API Key (for TTS)", type="password", value=OPENAI_API_KEY if OPENAI_API_KEY != "your_openai_api_key_here" else "")
            if openai_key_input:
                os.environ["OPENAI_API_KEY"] = openai_key_input
                globals()["OPENAI_API_KEY"] = openai_key_input
            
            st.session_state.tts_enabled = st.checkbox("Enable Text-to-Speech", value=st.session_state.get('tts_enabled', False))
            if st.session_state.tts_enabled:
                st.session_state.selected_voice = st.selectbox("Voice", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"], index=0)
                st.info("💡 If OpenAI API key is not provided, browser speech synthesis will be used as fallback.")
        
        # Test microphone button
        if st.button("🎤 Test Microphone", use_container_width=True):
            with st.spinner("Testing microphone..."):
                st.session_state.microphone_working = test_microphone()
            
            if st.session_state.microphone_working:
                st.success("✅ Microphone is working!")
            else:
                st.error("❌ Microphone test failed! Check your permissions and microphone.")
        
        st.markdown("---")
        
        # Single conversation
        if st.button("🗣️ Single Conversation", disabled=not st.session_state.microphone_working, use_container_width=True):
            handle_single_conversation()
        
        st.markdown("---")
        
        # Continuous listening controls
        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("▶️ Start Continuous", 
                        disabled=st.session_state.is_continuous_listening or not st.session_state.microphone_working,
                        use_container_width=True):
                st.session_state.is_continuous_listening = True
                st.session_state.current_listening_status = "Starting..."
                st.session_state.audio_processor.start_continuous_listening()
                st.rerun()
        
        with col1b:
            if st.button("⏹️ Stop Continuous", 
                        disabled=not st.session_state.is_continuous_listening,
                        use_container_width=True):
                st.session_state.is_continuous_listening = False
                st.session_state.audio_processor.stop_continuous_listening()
                st.rerun()
    
    with col2:
        st.subheader("📊 Status")
        
        # System status indicators
        status_container = st.container()
        with status_container:
            # Microphone status
            if st.session_state.microphone_working:
                st.success("🎤 Microphone: Ready")
            else:
                st.warning("🎤 Microphone: Not tested")
            
            # API status
            if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
                st.success("🔑 Groq API: Configured")
            else:
                st.error("🔑 Groq API: Not configured")
            
            # OpenAI API status
            if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
                st.success("🔑 OpenAI API: Configured")
            else:
                st.warning("🔑 OpenAI API: Not configured (TTS fallback will be used)")
            
            # Listening status
            if st.session_state.is_continuous_listening:
                st.success(f"🔴 Status: {st.session_state.current_listening_status}")
                
                # Show last transcription and response in continuous mode
                if st.session_state.last_transcription:
                    st.info(f"👤 Last heard: {st.session_state.last_transcription}")
                if st.session_state.last_ai_response:
                    st.info(f"🤖 Last response: {st.session_state.last_ai_response[:100]}...")
            else:
                st.info("⚪ Status: Idle")
            
            # TTS status
            if st.session_state.get('tts_enabled', False):
                st.success("🔊 TTS: Enabled")
            else:
                st.info("🔊 TTS: Disabled")
    
    with col3:
        st.subheader("🔊 Audio")
        # Audio playback from queue
        if not audio_queue.empty():
            try:
                audio_type, audio_data = audio_queue.get_nowait()
                if audio_type == "audio":
                    st.audio(audio_data, format='audio/mp3')
                elif audio_type == "html":
                    st.components.v1.html(audio_data, height=0)
            except queue.Empty:
                pass
        else:
            st.info("No audio to play")
    
    st.markdown("---")
    
    # Conversation history
    if st.session_state.conversation_history:
        st.subheader("💬 Conversation History")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.conversation_history = []
            st.rerun()
        
        # Show conversations in reverse order (newest first)
        for i, conv in enumerate(reversed(st.session_state.conversation_history[-10:])):
            idx = len(st.session_state.conversation_history) - i
            with st.expander(f"💬 Conversation {idx} - {conv['timestamp'].strftime('%H:%M:%S')}"):
                st.write(f"**👤 You:** {conv['user']}")
                st.write(f"**🤖 AI:** {conv['ai']}")
    else:
        st.info("No conversations yet. Start by testing your microphone and then try a single conversation!")
    
    # Debug information
    with st.expander("🔧 Debug Information", expanded=False):
        col_debug1, col_debug2 = st.columns(2)
        with col_debug1:
            if st.button("🗑️ Clear Debug Log"):
                st.session_state.debug_info = []
                st.rerun()
        
        with col_debug2:
            if st.button("🔄 Refresh Debug"):
                st.rerun()
        
        if st.session_state.debug_info:
            debug_text = "\n".join(st.session_state.debug_info[-50:])  # Show last 50 messages
            st.text_area("Debug Log", debug_text, height=300, disabled=True)
        else:
            st.info("No debug information available")
    
    # Auto-refresh when continuous listening is active (but less frequently)
    if st.session_state.is_continuous_listening:
        time.sleep(2)  # Increased interval to reduce resource usage
        st.rerun()

def cleanup():
    """Cleanup function to stop threads when app closes"""
    try:
        if hasattr(st.session_state, 'audio_processor') and st.session_state.audio_processor:
            st.session_state.audio_processor.stop_continuous_listening()
    except Exception as e:
        print(f"Cleanup error: {e}")

# Register cleanup function
atexit.register(cleanup)

if __name__ == "__main__":
    main()