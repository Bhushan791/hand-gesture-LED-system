
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import threading
import time
import os
import sys
import wave as wave_module

# Optional sound support
try:
    import pygame
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False
    print("pygame not available - sounds will be disabled")

class AdvancedFingerLEDSimulator:
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("🤚 Advanced Finger Gesture LED Simulator - Created by Bhushan")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0a0a0a')
        self.root.resizable(True, True)
        
        # Enhanced color scheme
        self.colors = {
            'bg_primary': '#0a0a0a',
            'bg_secondary': '#1a1a2e',
            'bg_card': '#16213e',
            'accent': '#00d4ff',
            'success': '#00ff88',
            'warning': '#ffaa00',
            'error': '#ff4757',
            'text_primary': '#ffffff',
            'text_secondary': '#b8c6db',
            'bulb_on': '#ffff00',
            'bulb_off': '#404040'
        }
        
        # Finger tracking variables
        self.finger_names = ["Thumb", "Index", "Middle", "Ring", "Little"]
        self.finger_states = [False] * 5
        self.previous_states = [False] * 5
        self.finger_confidence = [0.0] * 5
        self.detection_history = [[] for _ in range(5)]  # History for smoothing
        self.history_length = 5
        
        # Initialize MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Camera variables
        self.cap = None
        self.camera_running = False
        self.current_frame = None
        self.hand_landmarks = None
        self.frame_count = 0
        
        # GUI elements
        self.bulb_labels = []
        self.finger_labels = []
        self.confidence_bars = []
        self.status_labels = []
        self.camera_label = None
        self.info_text = None
        
        # Load or create bulb images
        self.bulb_on_image = None
        self.bulb_off_image = None
        self.load_bulb_images()
        
        # Sound system - Initialize FIRST
        self.setup_sound_system()
        
        # Create enhanced GUI
        self.setup_gui()
        
        # Start camera
        self.start_camera()
        
        # Start update loops
        self.update_gui_loop()
    
    def load_bulb_images(self):
        """Load bulb images from PNG files or create defaults"""
        try:
            # Try to load PNG images
            if os.path.exists("bulb_on.png"):
                on_img = Image.open("bulb_on.png").convert("RGBA")
                on_img = on_img.resize((100, 100), Image.Resampling.LANCZOS)
                self.bulb_on_image = ImageTk.PhotoImage(on_img)
                print("✅ Loaded bulb_on.png")
            else:
                self.bulb_on_image = self.create_default_bulb(True)
                print("⚠️ bulb_on.png not found, using default")
            
            if os.path.exists("bulb_off.png"):
                off_img = Image.open("bulb_off.png").convert("RGBA")
                off_img = off_img.resize((100, 100), Image.Resampling.LANCZOS)
                self.bulb_off_image = ImageTk.PhotoImage(off_img)
                print("✅ Loaded bulb_off.png")
            else:
                self.bulb_off_image = self.create_default_bulb(False)
                print("⚠️ bulb_off.png not found, using default")
                
        except Exception as e:
            print(f"Error loading images: {e}")
            self.bulb_on_image = self.create_default_bulb(True)
            self.bulb_off_image = self.create_default_bulb(False)
    
    def create_default_bulb(self, is_on=True):
        """Create default bulb image"""
        size = 100
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        center = size // 2
        radius = size // 2 - 10
        
        if is_on:
            # Create glowing ON bulb
            for i in range(radius + 15, 0, -2):
                if i > radius:
                    alpha = max(0, int(60 * ((radius + 15 - i) / 15)))
                    draw.ellipse([center - i, center - i, center + i, center + i], 
                                fill=(255, 255, 0, alpha))
                else:
                    alpha = 255
                    yellow_val = int(255 * (i / radius))
                    draw.ellipse([center - i, center - i, center + i, center + i], 
                                fill=(255, yellow_val, 0, alpha))
            
            # Bright center
            draw.ellipse([center - radius//3, center - radius//3, 
                         center + radius//3, center + radius//3], 
                        fill=(255, 255, 200, 255))
        else:
            # Create OFF bulb
            for i in range(radius, 0, -2):
                gray_val = int(40 + 60 * (i / radius))
                alpha = int(150 * (i / radius))
                draw.ellipse([center - i, center - i, center + i, center + i], 
                            fill=(gray_val, gray_val, gray_val, alpha))
        
        return ImageTk.PhotoImage(img)
    
    def setup_sound_system(self):
        """Initialize sound system exactly like original code"""
        self.sounds_enabled = False
        self.sound_on = None
        self.sound_off = None
        
        if SOUND_AVAILABLE:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                
                # Create default sounds if files don't exist (like original code)
                if not os.path.exists("sound_on.wav"):
                    self.create_default_sounds()
                
                # Load sound files
                if os.path.exists("sound_on.wav") and os.path.exists("sound_off.wav"):
                    self.sound_on = pygame.mixer.Sound("sound_on.wav")
                    self.sound_off = pygame.mixer.Sound("sound_off.wav")
                    self.sounds_enabled = True
                    print("✅ Sound system initialized with WAV files")
                else:
                    print("⚠️ Sound files not found, creating defaults...")
                    self.create_default_sounds()
                    if os.path.exists("sound_on.wav") and os.path.exists("sound_off.wav"):
                        self.sound_on = pygame.mixer.Sound("sound_on.wav")
                        self.sound_off = pygame.mixer.Sound("sound_off.wav")
                        self.sounds_enabled = True
                        print("✅ Sound system initialized with generated sounds")
                    
            except Exception as e:
                print(f"Sound initialization failed: {e}")
                self.sounds_enabled = False
    
    def create_default_sounds(self):
        """Create default sound files exactly like original code"""
        try:
            import numpy as np
            
            # Create ON sound (ascending beep) - exactly like original
            sample_rate = 22050
            duration = 0.2
            t = np.linspace(0, duration, int(sample_rate * duration))
            frequency_start = 800
            frequency_end = 1200
            frequency = np.linspace(frequency_start, frequency_end, len(t))
            wave = np.sin(2 * np.pi * frequency * t) * np.exp(-t * 5)
            wave = (wave * 32767).astype(np.int16)
            
            # Save ON sound
            with wave_module.open("sound_on.wav", "w") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(wave.tobytes())
            
            # Create OFF sound (descending beep) - exactly like original
            frequency = np.linspace(frequency_end, frequency_start, len(t))
            wave = np.sin(2 * np.pi * frequency * t) * np.exp(-t * 8)
            wave = (wave * 16383).astype(np.int16)  # Quieter
            
            # Save OFF sound
            with wave_module.open("sound_off.wav", "w") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(wave.tobytes())
            
            print("✅ Default sound files created successfully")
                
        except Exception as e:
            print(f"Could not create default sounds: {e}")
            self.sounds_enabled = False
    
    def setup_gui(self):
        """Create the enhanced GUI"""
        # Header
        header_frame = tk.Frame(self.root, bg=self.colors['bg_primary'], height=80)
        header_frame.pack(fill='x', padx=20, pady=(20, 10))
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(header_frame, 
                              text="🤚 Advanced Finger Gesture LED Simulator",
                              font=('Arial', 24, 'bold'),
                              fg=self.colors['accent'],
                              bg=self.colors['bg_primary'])
        title_label.pack(side='left', pady=20)
        
        creator_label = tk.Label(header_frame,
                               text="Created by: Bhushan",
                               font=('Arial', 14, 'italic'),
                               fg=self.colors['text_secondary'],
                               bg=self.colors['bg_primary'])
        creator_label.pack(side='right', pady=20)
        
        # Main content
        main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Camera
        left_panel = tk.Frame(main_frame, bg=self.colors['bg_secondary'], 
                             relief='solid', borderwidth=2)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        self.setup_camera_panel(left_panel)
        
        # Right panel - LEDs and Info
        right_panel = tk.Frame(main_frame, bg=self.colors['bg_secondary'],
                              relief='solid', borderwidth=2)
        right_panel.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        self.setup_led_panel(right_panel)
        self.setup_info_panel(right_panel)
        
        # Bottom controls
        self.setup_controls(self.root)
    
    def setup_camera_panel(self, parent):
        """Setup camera display"""
        camera_frame = tk.LabelFrame(parent, text="📹 Live Camera Feed", 
                                   font=('Arial', 16, 'bold'),
                                   fg=self.colors['text_primary'],
                                   bg=self.colors['bg_secondary'],
                                   labelanchor='n')
        camera_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Camera display
        self.camera_label = tk.Label(camera_frame, 
                                   text="Initializing camera...",
                                   font=('Arial', 16),
                                   fg=self.colors['text_secondary'],
                                   bg=self.colors['bg_card'],
                                   width=50, height=20)
        self.camera_label.pack(expand=True, fill='both', padx=15, pady=15)
        
        # Status bar
        status_frame = tk.Frame(camera_frame, bg=self.colors['bg_secondary'])
        status_frame.pack(fill='x', padx=15, pady=(0, 15))
        
        self.camera_status = tk.Label(status_frame,
                                    text="🔴 Camera: Starting...",
                                    font=('Arial', 12, 'bold'),
                                    fg=self.colors['warning'],
                                    bg=self.colors['bg_secondary'])
        self.camera_status.pack(side='left')
        
        self.fps_label = tk.Label(status_frame,
                                text="FPS: --",
                                font=('Arial', 12),
                                fg=self.colors['text_secondary'],
                                bg=self.colors['bg_secondary'])
        self.fps_label.pack(side='right')
    
    def setup_led_panel(self, parent):
        """Setup LED display panel"""
        led_frame = tk.LabelFrame(parent, text="💡 LED Finger Control", 
                                font=('Arial', 16, 'bold'),
                                fg=self.colors['text_primary'],
                                bg=self.colors['bg_secondary'],
                                labelanchor='n')
        led_frame.pack(fill='x', padx=15, pady=15)
        
        # Instructions
        instruction_label = tk.Label(led_frame,
                                   text="Make a fist = All OFF | Raise fingers = LEDs ON",
                                   font=('Arial', 12, 'italic'),
                                   fg=self.colors['text_secondary'],
                                   bg=self.colors['bg_secondary'])
        instruction_label.pack(pady=(10, 5))
        
        # LED container
        led_container = tk.Frame(led_frame, bg=self.colors['bg_card'])
        led_container.pack(fill='x', padx=20, pady=20)
        
        # Create LED displays
        for i in range(5):
            finger_frame = tk.Frame(led_container, bg=self.colors['bg_card'])
            finger_frame.grid(row=0, column=i, padx=10, pady=15)
            
            # LED bulb
            bulb_label = tk.Label(finger_frame, bg=self.colors['bg_card'],
                                image=self.bulb_off_image)
            bulb_label.pack()
            self.bulb_labels.append(bulb_label)
            
            # Finger name
            name_label = tk.Label(finger_frame, 
                                text=self.finger_names[i],
                                font=('Arial', 11, 'bold'),
                                fg=self.colors['text_primary'],
                                bg=self.colors['bg_card'])
            name_label.pack(pady=(8, 2))
            self.finger_labels.append(name_label)
            
            # Status
            status_label = tk.Label(finger_frame,
                                  text="OFF",
                                  font=('Arial', 10, 'bold'),
                                  fg=self.colors['bulb_off'],
                                  bg=self.colors['bg_card'])
            status_label.pack(pady=2)
            self.status_labels.append(status_label)
            
            # Confidence indicator
            conf_frame = tk.Frame(finger_frame, bg=self.colors['bg_card'])
            conf_frame.pack(fill='x', pady=(5, 0))
            
            conf_canvas = tk.Canvas(conf_frame, width=70, height=6, 
                                  bg=self.colors['bg_primary'],
                                  highlightthickness=0)
            conf_canvas.pack()
            self.confidence_bars.append(conf_canvas)
    
    def setup_info_panel(self, parent):
        """Setup information panel"""
        info_frame = tk.LabelFrame(parent, text="📊 Detection Information", 
                                 font=('Arial', 16, 'bold'),
                                 fg=self.colors['text_primary'],
                                 bg=self.colors['bg_secondary'],
                                 labelanchor='n')
        info_frame.pack(fill='both', expand=True, padx=15, pady=(0, 15))
        
        # Info display
        text_frame = tk.Frame(info_frame)
        text_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        self.info_text = tk.Text(text_frame,
                               font=('Consolas', 9),
                               bg=self.colors['bg_card'],
                               fg=self.colors['text_secondary'],
                               height=12,
                               wrap='word',
                               state='normal')
        
        scrollbar = tk.Scrollbar(text_frame, orient='vertical')
        scrollbar.pack(side='right', fill='y')
        
        self.info_text.pack(side='left', fill='both', expand=True)
        self.info_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.info_text.yview)
        
        # Initial info
        self.log_info("🚀 Advanced Finger LED Simulator Started")
        self.log_info("👨‍💻 Created by: Bhushan")
        self.log_info("=" * 45)
        self.log_info("📋 How to use:")
        self.log_info("  • Position hand clearly in camera view")
        self.log_info("  • Start with closed fist (all LEDs OFF)")
        self.log_info("  • Raise individual fingers to turn ON LEDs")
        self.log_info("  • Each finger controls one LED bulb")
        self.log_info("  • Works from both open→closed and closed→open")
        self.log_info("=" * 45)
        
        # Log sound status
        if self.sounds_enabled:
            self.log_info("🔊 Sound system: ENABLED")
        else:
            self.log_info("🔇 Sound system: DISABLED")
    
    def setup_controls(self, parent):
        """Setup control buttons"""
        control_frame = tk.Frame(parent, bg=self.colors['bg_primary'])
        control_frame.pack(fill='x', padx=20, pady=(10, 20))
        
        btn_frame = tk.Frame(control_frame, bg=self.colors['bg_primary'])
        btn_frame.pack()
        
        # Test button
        test_btn = tk.Button(btn_frame,
                           text="🧪 Test All LEDs",
                           command=self.test_all_leds,
                           font=('Arial', 12, 'bold'),
                           bg=self.colors['success'],
                           fg='black',
                           padx=25, pady=12,
                           relief='raised',
                           borderwidth=2)
        test_btn.pack(side='left', padx=15)
        
        # Sound toggle
        self.sound_btn = tk.Button(btn_frame,
                                 text="🔊 Sound: ON" if self.sounds_enabled else "🔇 Sound: OFF",
                                 command=self.toggle_sound,
                                 font=('Arial', 12, 'bold'),
                                 bg=self.colors['accent'],
                                 fg='black',
                                 padx=25, pady=12,
                                 relief='raised',
                                 borderwidth=2)
        self.sound_btn.pack(side='left', padx=15)
        
        # Reset button
        reset_btn = tk.Button(btn_frame,
                            text="🔄 Reset",
                            command=self.reset_detection,
                            font=('Arial', 12, 'bold'),
                            bg=self.colors['warning'],
                            fg='black',
                            padx=25, pady=12,
                            relief='raised',
                            borderwidth=2)
        reset_btn.pack(side='left', padx=15)
        
        # Quit button
        quit_btn = tk.Button(btn_frame,
                           text="❌ Exit",
                           command=self.quit_app,
                           font=('Arial', 12, 'bold'),
                           bg=self.colors['error'],
                           fg='white',
                           padx=25, pady=12,
                           relief='raised',
                           borderwidth=2)
        quit_btn.pack(side='left', padx=15)
    
    def start_camera(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(0)     ###for exxternal webcam
            if not self.cap.isOpened():
                raise Exception("Cannot access camera")
            
            # Optimize camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.camera_running = True
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            self.log_info("✅ Camera initialized successfully")
            
        except Exception as e:
            self.log_info(f"❌ Camera error: {e}")
            messagebox.showerror("Camera Error", f"Failed to start camera: {e}")
    
    def camera_loop(self):
        """Main camera processing loop"""
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.camera_running and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Mirror effect
                frame = cv2.flip(frame, 1)
                self.current_frame = frame.copy()
                
                # Process hand detection
                self.process_hand_detection(frame)
                
                # Update FPS counter
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    fps = fps_counter / (time.time() - fps_start_time)
                    self.root.after(0, lambda: self.fps_label.configure(text=f"FPS: {fps:.1f}"))
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Display frame
                self.display_frame(frame)
                
                self.frame_count += 1
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                self.log_info(f"⚠️ Camera processing error: {e}")
                time.sleep(0.1)
    
    def process_hand_detection(self, frame):
        """Enhanced hand detection with improved finger logic"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        new_states = [False] * 5
        confidences = [0.0] * 5
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.hand_landmarks = hand_landmarks
                
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Extract landmarks
                landmarks = self.extract_landmarks(hand_landmarks, frame.shape)
                
                # Detect fingers with improved algorithm
                new_states, confidences = self.detect_fingers_advanced(landmarks)
                
                # Update detection history for smoothing
                self.update_detection_history(new_states, confidences)
                
                # Apply smoothing
                new_states, confidences = self.apply_smoothing()
                
                break  # Only process first hand
        else:
            self.hand_landmarks = None
            # Clear detection history when no hand detected
            self.detection_history = [[] for _ in range(5)]
        
        # Update finger states
        self.update_finger_states(new_states, confidences)
    
    def extract_landmarks(self, hand_landmarks, frame_shape):
        """Extract landmark coordinates"""
        h, w, _ = frame_shape
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([int(lm.x * w), int(lm.y * h), lm.z])
        return landmarks
    
    def detect_fingers_advanced(self, landmarks):
        """Advanced finger detection algorithm"""
        finger_states = [False] * 5
        confidences = [0.0] * 5
        
        # Landmark indices
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [3, 6, 10, 14, 18]
        mcp_ids = [2, 5, 9, 13, 17]
        
        try:
            # Get wrist position for reference
            wrist = landmarks[0]
            
            for i in range(5):
                tip = landmarks[tip_ids[i]]
                pip = landmarks[pip_ids[i]]
                mcp = landmarks[mcp_ids[i]]
                
                if i == 0:  # Thumb - horizontal logic
                    # Check thumb extension based on distance from wrist
                    tip_wrist_dist = self.calculate_distance(tip, wrist)
                    mcp_wrist_dist = self.calculate_distance(mcp, wrist)
                    
                    # Thumb is extended if tip is significantly farther from wrist than MCP
                    extension_ratio = tip_wrist_dist / max(mcp_wrist_dist, 1)
                    
                    if extension_ratio > 1.3:  # Threshold for thumb extension
                        finger_states[i] = True
                        confidences[i] = min(1.0, (extension_ratio - 1.3) * 2)
                    
                else:  # Other fingers - vertical logic with multiple checks
                    # Check 1: Tip above PIP and MCP
                    tip_above_pip = tip[1] < pip[1]
                    tip_above_mcp = tip[1] < mcp[1]
                    
                    # Check 2: Significant vertical separation
                    pip_tip_dist = pip[1] - tip[1]
                    mcp_pip_dist = abs(mcp[1] - pip[1])
                    
                    # Check 3: Finger straightness (tip should be roughly aligned)
                    alignment_score = self.calculate_finger_alignment(tip, pip, mcp)
                    
                    # Check 4: Distance from wrist (extended fingers are farther)
                    tip_wrist_dist = self.calculate_distance(tip, wrist)
                    mcp_wrist_dist = self.calculate_distance(mcp, wrist)
                    extension_ratio = tip_wrist_dist / max(mcp_wrist_dist, 1)
                    
                    # Combine all checks
                    if (tip_above_pip and tip_above_mcp and 
                        pip_tip_dist > 15 and  # Minimum extension distance
                        alignment_score > 0.6 and  # Reasonable alignment
                        extension_ratio > 1.1):  # Extended from wrist
                        
                        finger_states[i] = True
                        # Calculate confidence based on multiple factors
                        conf_factors = [
                            min(1.0, pip_tip_dist / 30),  # Extension distance
                            alignment_score,  # Finger straightness
                            min(1.0, (extension_ratio - 1.1) * 2)  # Extension ratio
                        ]
                        confidences[i] = sum(conf_factors) / len(conf_factors)
                    
        except (IndexError, ZeroDivisionError) as e:
            self.log_info(f"Detection error: {e}")
        
        return finger_states, confidences
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5
    
    def calculate_finger_alignment(self, tip, pip, mcp):
        """Calculate how straight the finger is (0-1 score)"""
        try:
            # Vector from MCP to PIP
            vec1 = [pip[0] - mcp[0], pip[1] - mcp[1]]
            # Vector from PIP to TIP
            vec2 = [tip[0] - pip[0], tip[1] - pip[1]]
            
            # Calculate dot product and magnitudes
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            mag1 = (vec1[0]**2 + vec1[1]**2)**0.5
            mag2 = (vec2[0]**2 + vec2[1]**2)**0.5
            
            if mag1 == 0 or mag2 == 0:
                return 0
            
            # Cosine of angle between vectors
            cos_angle = dot_product / (mag1 * mag2)
            cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
            
            # Convert to alignment score (1 = perfectly aligned, 0 = perpendicular)
            alignment = (cos_angle + 1) / 2
            return alignment
            
        except:
            return 0
    
    def update_detection_history(self, states, confidences):
        """Update detection history for smoothing"""
        for i in range(5):
            self.detection_history[i].append((states[i], confidences[i]))
            if len(self.detection_history[i]) > self.history_length:
                self.detection_history[i].pop(0)
    
    def apply_smoothing(self):
        """Apply temporal smoothing to reduce jitter"""
        smoothed_states = [False] * 5
        smoothed_confidences = [0.0] * 5
        
        for i in range(5):
            if len(self.detection_history[i]) > 0:
                # Count positive detections in history
                positive_count = sum(1 for state, _ in self.detection_history[i] if state)
                total_count = len(self.detection_history[i])
                
                # Average confidence
                avg_confidence = sum(conf for _, conf in self.detection_history[i]) / total_count
                
                # Finger is considered "up" if majority of recent detections are positive
                threshold = 0.6 if total_count >= 3 else 0.5
                smoothed_states[i] = (positive_count / total_count) >= threshold
                smoothed_confidences[i] = avg_confidence
        
        return smoothed_states, smoothed_confidences
    
    def update_finger_states(self, new_states, confidences):
        """Update finger states with sound feedback - FIXED VERSION"""
        # Check for state changes and play sounds
        if self.sounds_enabled and self.sound_on is not None and self.sound_off is not None:
            for i in range(5):
                if new_states[i] != self.finger_states[i]:
                    try:
                        if new_states[i]:  # Finger went UP
                            self.sound_on.play()
                            self.log_info(f"🔊 {self.finger_names[i]} finger UP - Sound played")
                        else:  # Finger went DOWN
                            self.sound_off.play()
                            self.log_info(f"🔇 {self.finger_names[i]} finger DOWN - Sound played")
                    except Exception as e:
                        self.log_info(f"Sound error: {e}")
        
        # Update states
        self.previous_states = self.finger_states.copy()
        self.finger_states = new_states.copy()
        self.finger_confidence = confidences.copy()
        
        # Update GUI
        self.root.after(0, self.update_led_display)
        
        # Log significant changes
        fingers_up = sum(new_states)
        prev_fingers_up = sum(self.previous_states)
        if fingers_up != prev_fingers_up:
            self.log_info(f"🤚 Total fingers detected: {fingers_up}/5")
    
    def update_led_display(self):
        """Update LED display and indicators"""
        for i in range(5):
            # Update bulb image
            if self.finger_states[i]:
                self.bulb_labels[i].configure(image=self.bulb_on_image)
                self.status_labels[i].configure(text="ON", fg=self.colors['success'])
                self.finger_labels[i].configure(fg=self.colors['success'])
            else:
                self.bulb_labels[i].configure(image=self.bulb_off_image)
                self.status_labels[i].configure(text="OFF", fg=self.colors['bulb_off'])
                self.finger_labels[i].configure(fg=self.colors['text_primary'])
            
            # Update confidence bar
            self.update_confidence_bar(i, self.finger_confidence[i])
    
    def update_confidence_bar(self, finger_idx, confidence):
        """Update confidence visualization"""
        canvas = self.confidence_bars[finger_idx]
        canvas.delete("all")
        
        # Background
        canvas.create_rectangle(0, 0, 70, 6, fill=self.colors['bg_primary'], outline="")
        
        # Confidence bar
        width = int(70 * confidence)
        if confidence > 0.8:
            color = self.colors['success']
        elif confidence > 0.5:
            color = self.colors['warning']
        else:
            color = self.colors['error']
        
        if width > 0:
            canvas.create_rectangle(0, 0, width, 6, fill=color, outline="")
    
    def display_frame(self, frame):
        """Display camera frame in GUI"""
        try:
            # Resize for display
            display_frame = cv2.resize(frame, (480, 360))
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PhotoImage
            pil_image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update display
            self.camera_label.configure(image=photo)
            self.camera_label.image = photo
            
            # Update status
            if self.hand_landmarks:
                self.camera_status.configure(text="🟢 Camera: Hand Detected",
                                           fg=self.colors['success'])
            else:
                self.camera_status.configure(text="🟡 Camera: No Hand Detected",
                                           fg=self.colors['warning'])
                
        except Exception as e:
            self.log_info(f"Display error: {e}")
    
    def update_gui_loop(self):
        """Main GUI update loop"""
        try:
            self.root.after(50, self.update_gui_loop)
        except:
            pass
    
    def test_all_leds(self):
        """Test all LEDs in sequence with sounds"""
        def test_sequence():
            original_states = self.finger_states.copy()
            
            self.log_info("🧪 Starting LED test sequence...")
            
            # Turn on each LED individually with sound
            for i in range(5):
                self.finger_states = [False] * 5
                self.finger_states[i] = True
                self.finger_confidence = [0.0] * 5
                self.finger_confidence[i] = 1.0
                self.root.after(0, self.update_led_display)
                
                # Play sound
                if self.sounds_enabled and self.sound_on:
                    try:
                        self.sound_on.play()
                    except:
                        pass
                
                time.sleep(0.3)
            
            # Turn on all LEDs
            self.finger_states = [True] * 5
            self.finger_confidence = [1.0] * 5
            self.root.after(0, self.update_led_display)
            time.sleep(0.5)
            
            # Turn off all LEDs with sound
            self.finger_states = [False] * 5
            self.finger_confidence = [0.0] * 5
            self.root.after(0, self.update_led_display)
            
            if self.sounds_enabled and self.sound_off:
                try:
                    self.sound_off.play()
                except:
                    pass
            
            time.sleep(0.3)
            
            # Restore original states
            self.finger_states = original_states
            self.root.after(0, self.update_led_display)
            
            self.log_info("✅ LED test completed")
        
        threading.Thread(target=test_sequence, daemon=True).start()
    
    def toggle_sound(self):
        """Toggle sound effects"""
        if SOUND_AVAILABLE and self.sound_on is not None and self.sound_off is not None:
            self.sounds_enabled = not self.sounds_enabled
            self.sound_btn.configure(
                text="🔊 Sound: ON" if self.sounds_enabled else "🔇 Sound: OFF"
            )
            self.log_info(f"🔊 Sound {'enabled' if self.sounds_enabled else 'disabled'}")
            
            # Test sound when enabling
            if self.sounds_enabled:
                try:
                    self.sound_on.play()
                    self.log_info("🔊 Sound test played")
                except Exception as e:
                    self.log_info(f"Sound test failed: {e}")
        else:
            self.log_info("🔇 Sound system not available")
    
    def reset_detection(self):
        """Reset detection system"""
        self.detection_history = [[] for _ in range(5)]
        self.finger_states = [False] * 5
        self.finger_confidence = [0.0] * 5
        self.update_led_display()
        self.log_info("🔄 Detection system reset")
    
    def log_info(self, message):
        """Log information with timestamp"""
        if self.info_text:
            timestamp = time.strftime("%H:%M:%S")
            self.info_text.insert('end', f"[{timestamp}] {message}\n")
            self.info_text.see('end')
            
            # Limit text length
            lines = self.info_text.get("1.0", "end").split('\n')
            if len(lines) > 100:
                self.info_text.delete("1.0", "10.0")
    
    def quit_app(self):
        """Gracefully quit application"""
        self.log_info("🛑 Shutting down application...")
        self.camera_running = False
        
        if self.cap is not None:
            self.cap.release()
        
        if hasattr(self, 'camera_thread'):
            self.camera_thread.join(timeout=2.0)
        
        if SOUND_AVAILABLE:
            try:
                pygame.mixer.quit()
            except:
                pass
        
        self.root.quit()
        self.root.destroy()
    
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.quit_app()

def main():
    """Main function"""
    print("🤚 Advanced Finger Gesture LED Simulator")
    print("Created by: Bhushan")
    print("=" * 50)
    
    # Check dependencies
    required_modules = ['cv2', 'mediapipe', 'PIL', 'numpy']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing dependencies: {', '.join(missing_modules)}")
        print("Install with: pip install opencv-python mediapipe pillow numpy pygame")
        sys.exit(1)
    
    print("✅ All dependencies found")
    print("🚀 Starting application...")
    
    # Create and run application
    app = AdvancedFingerLEDSimulator()
    app.run()

if __name__ == "__main__":
    main()
