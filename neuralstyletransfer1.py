import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import os
import time
import threading
from queue import Queue
import argparse
from pathlib import Path

class StyleTransferApp:
    def __init__(self, model_size=256, camera_id=0, frame_skip=0):
        """
        Initialize the Style Transfer Application
        
        Args:
            model_size: Size for style transfer model (256)
            camera_id: Camera device ID
            frame_skip: Process every N frames for better performance
        """
        self.model_size = model_size
        self.camera_id = camera_id
        self.frame_skip = frame_skip
        
        #Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        #Threading for better performance
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.processing_thread = None
        self.running = False
        
        #Style management
        self.current_style = 0
        self.style_tensors = []
        self.style_names = []
        
        #UI settings
        self.show_original = True
        self.show_fps = True
        self.show_help = False
        self.frame_counter = 0
        
        #Performance measurement
        self.inference_times = []
        self.total_frames_processed = 0
        self.latency_samples = []
        
        self.setup_gpu()
        self.load_model()
        self.load_style_images()
        
    def setup_gpu(self):
        #Configure GPU settings for optimal performance
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU acceleration enabled with {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("No GPU found, using CPU (may be slower)")
    
    def load_model(self):
        #Load the TensorFlow Hub style transfer model
        hub_handle = f'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-{self.model_size}/2'
        print(f"Loading TensorFlow Hub model (size: {self.model_size})...")
        
        try:
            self.hub_module = hub.load(hub_handle)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def load_style_images(self):
        #Load and preprocess style images
        style_urls = {
            'Van Gogh - Starry Night': 'https://upload.wikimedia.org/wikipedia/commons/c/cd/VanGogh-starry_night.jpg',
            'Picasso - Les Demoiselles': 'https://upload.wikimedia.org/wikipedia/commons/2/2a/Les_Demoiselles_d%27Avignon_%287925004644%29.jpg',
            'Monet - Impression Sunrise': 'https://upload.wikimedia.org/wikipedia/commons/5/54/Claude_Monet%2C_Impression%2C_soleil_levant.jpg',
            'Hokusai - Great Wave': 'https://upload.wikimedia.org/wikipedia/commons/0/0d/Great_Wave_off_Kanagawa2.jpg',
            'Kandinsky - Composition VII': 'https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
        }
        
        print("Loading style images...")
        
        for name, url in style_urls.items():
            try:
                #Download image
                local_path = tf.keras.utils.get_file(
                    os.path.basename(url).split('?')[0],  #Remove URL parameters
                    url,
                    cache_subdir='style_images'
                )
                
                #Load and preprocess
                img = tf.io.read_file(local_path)
                img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
                img = tf.image.resize(img, (self.model_size, self.model_size))
                img = img[tf.newaxis, ...]  #Add batch dimension
                
                self.style_tensors.append(img)
                self.style_names.append(name)
                print(f"Loaded: {name}")
                
            except Exception as e:
                print(f"Failed to load {name}: {e}")
        
        if not self.style_tensors:
            raise RuntimeError("No style images loaded successfully")
        
        print(f"Total styles loaded: {len(self.style_names)}")
    
    def preprocess_frame(self, frame):
        #Preprocess camera frame for style transfer
        #Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Resize for faster processing while maintaining aspect ratio
        h, w = rgb.shape[:2]
        target_size = 256  #Balanced size for speed vs quality
        
        if max(h, w) > target_size:
            if h > w:
                new_h, new_w = target_size, int(w * target_size / h)
            else:
                new_h, new_w = int(h * target_size / w), target_size
            rgb = cv2.resize(rgb, (new_w, new_h))
        
        #Normalize to [0, 1]
        rgb = rgb.astype(np.float32) / 255.0
        
        return tf.constant(rgb[np.newaxis, ...])  #Add batch dimension
    
    def apply_style_transfer(self, content_tensor):
        #Apply style transfer to content tensor
        try:
            #Apply style transfer with timing
            start_time = time.time()
            outputs = self.hub_module(
                content_tensor,
                self.style_tensors[self.current_style]
            )
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            print(f"Inference time: {inference_time:.4f} sec")
            
            stylized = outputs[0][0]  #Remove batch dimension
            
            #Clip values and convert to uint8
            stylized = tf.clip_by_value(stylized, 0.0, 1.0)
            stylized_img = (stylized.numpy() * 255).astype(np.uint8)
            
            #Convert RGB back to BGR for OpenCV
            return cv2.cvtColor(stylized_img, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"Style transfer error: {e}")
            return None
    
    def processing_worker(self):
        #Worker thread for style transfer processing
        while self.running:
            try:
                if not self.frame_queue.empty():
                    item = self.frame_queue.get_nowait()
                    
                    #Unpack frame and timestamp for latency measurement
                    if isinstance(item, tuple):
                        frame, frame_timestamp = item
                    else:
                        frame, frame_timestamp = item, None
                    
                    #Preprocess frame
                    content_tensor = self.preprocess_frame(frame)
                    
                    #Apply style transfer
                    stylized_frame = self.apply_style_transfer(content_tensor)
                    
                    if stylized_frame is not None:
                        #Resize back to original frame size
                        original_h, original_w = frame.shape[:2]
                        stylized_frame = cv2.resize(stylized_frame, (original_w, original_h))
                        
                        #Track end-to-end latency
                        if frame_timestamp is not None:
                            latency = time.time() - frame_timestamp
                            self.latency_samples.append(latency)
                            print(f"End-to-End Latency: {latency:.4f} sec")
                        
                        #Track total stylized frames
                        self.total_frames_processed += 1
                        
                        #Put result in queue
                        if not self.result_queue.full():
                            self.result_queue.put(stylized_frame)
                
                time.sleep(0.001)  #Small delay to prevent CPU spinning
                
            except Exception as e:
                print(f"Processing worker error: {e}")
    
    def update_fps(self):
        #Update FPS counter
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_ui(self, frame, stylized_frame=None):
        #Draw UI elements on frame
        h, w = frame.shape[:2]
        
        #Draw style information
        style_text = f"Style: {self.style_names[self.current_style]} ({self.current_style + 1}/{len(self.style_names)})"
        cv2.putText(frame, style_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        #Draw FPS
        if self.show_fps:
            fps_text = f"FPS: {self.current_fps}"
            cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        #Draw controls
        if self.show_help:
            controls = [
            "Controls:",
            "1-5: Switch styles",
            "SPACE: Toggle original/stylized",
            "F: Toggle FPS display",
            "H: Toggle this help",
            "S: Save screenshot",
            "Q: Quit"
            ]
            for i, control in enumerate(controls):
                y_pos = h - 160 + i * 22
                cv2.putText(frame, control, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "H: help", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        
        return frame
    
    def save_screenshot(self, original_frame, stylized_frame):
        #Save screenshot of current frames
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        style_name = self.style_names[self.current_style].replace(' ', '_').replace('-', '_')
        
        #Create screenshots directory
        screenshots_dir = Path('screenshots')
        screenshots_dir.mkdir(exist_ok=True)
        
        #Save original
        original_path = screenshots_dir / f"original_{timestamp}.jpg"
        cv2.imwrite(str(original_path), original_frame)
        
        #Save stylized
        if stylized_frame is not None:
            stylized_path = screenshots_dir / f"stylized_{style_name}_{timestamp}.jpg"
            cv2.imwrite(str(stylized_path), stylized_frame)
        
        print(f"Screenshots saved: {original_path.name}")
    
    def run(self):
        #Main application loop
        #Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.camera_id}")
        
        #Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\nStyle Transfer Application Started!")
        print(f"Camera: {self.camera_id} | Model Size: {self.model_size}")
        print("Press 'H' for controls")
        
        #Start processing thread
        self.running = True
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        last_stylized_frame = None
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                #Update FPS counter
                self.update_fps()
                
                #Add frame to processing queue (skip frames for performance)
                #FIX: frame_skip=0 means process every frame (no division by zero)
                if self.frame_skip == 0:
                    should_process = True
                else:
                    should_process = (self.frame_counter % self.frame_skip == 0)
                
                if should_process:
                    if not self.frame_queue.full():
                        frame_timestamp = time.time()
                        self.frame_queue.put((frame.copy(), frame_timestamp))
                
                self.frame_counter += 1
                
                #Get processed frame if available
                if not self.result_queue.empty():
                    last_stylized_frame = self.result_queue.get_nowait()
                
                #Prepare display frame
                if self.show_original:
                    display_frame = self.draw_ui(frame.copy())
                    window_title = 'Original Feed'
                else:
                    if last_stylized_frame is not None:
                        display_frame = self.draw_ui(last_stylized_frame.copy())
                        window_title = 'Stylized Feed'
                    else:
                        display_frame = self.draw_ui(frame.copy())
                        window_title = 'Processing...'
                
                #Show frame
                cv2.imshow(window_title, display_frame)
                
                #Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):  #Space bar
                    self.show_original = not self.show_original
                elif key == ord('f'):
                    self.show_fps = not self.show_fps
                elif key == ord('h'):
                    self.show_help = not self.show_help 
                elif key == ord('s'):
                    self.save_screenshot(frame, last_stylized_frame)
                elif key >= ord('1') and key <= ord('5'):
                    style_index = key - ord('1')
                    if style_index < len(self.style_names):
                        self.current_style = style_index
                        print(f"Switched to: {self.style_names[self.current_style]}")
        
        except KeyboardInterrupt:
            print("\nStopped by user")
        
        finally:
            self.running = False
            if self.processing_thread:
                self.processing_thread.join(timeout=1)
            cap.release()
            cv2.destroyAllWindows()
            
            #Print performance summary
            if self.inference_times:
                avg_time = sum(self.inference_times) / len(self.inference_times)
                min_time = min(self.inference_times)
                max_time = max(self.inference_times)
                print("\n===== Performance Summary =====")
                print(f"Total Stylized Frames  : {self.total_frames_processed}")
                print(f"Average Inference Time : {avg_time:.4f} sec")
                print(f"Min Inference Time     : {min_time:.4f} sec")
                print(f"Max Inference Time     : {max_time:.4f} sec")
                print(f"True Stylization FPS   : {1/avg_time:.2f}")
                if self.latency_samples:
                    avg_latency = sum(self.latency_samples) / len(self.latency_samples)
                    print(f"Avg End-to-End Latency : {avg_latency:.4f} sec")
                print("================================")
            
            print("Cleanup completed")

def main():
    parser = argparse.ArgumentParser(description='Real-time Style Transfer Application')
    parser.add_argument('--model-size', type=int, choices=[256], default=256,
                        help='Model size (256 for speed)')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--frame-skip', type=int, default=2,
                        help='Process every N frames (higher = faster but choppier)')
    
    args = parser.parse_args()
    
    try:
        app = StyleTransferApp(
            model_size=args.model_size,
            camera_id=args.camera_id,
            frame_skip=args.frame_skip
        )
        app.run()
    except Exception as e:
        print(f"Application error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())