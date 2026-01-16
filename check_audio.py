import sounddevice as sd
import numpy as np
import time

def list_devices():
    print("Available Audio Devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"{i}: {dev['name']} (Channels: {dev['max_input_channels']}, SR: {dev['default_samplerate']})")
    print("-" * 30)
    return devices

def test_microphone(device_index=None):
    print(f"Testing device index: {device_index if device_index is not None else 'Default'}")
    
    duration = 5  # seconds
    print(f"Recording for {duration} seconds... Please speak or make noise.")
    
    try:
        def callback(indata, frames, time, status):
            if status:
                print(f"Status: {status}")
            rms = np.sqrt(np.mean(indata**2))
            peak = np.max(np.abs(indata))
            # Print a bar indicating volume
            dashes = int(peak * 50)
            print(f"RMS: {rms:.4f} | Peak: {peak:.4f} | {'#' * dashes}", end='\r')

        with sd.InputStream(device=device_index, channels=1, callback=callback):
            time.sleep(duration)
        print("\nFinished testing.")
        
    except Exception as e:
        print(f"\nError testing device: {e}")

if __name__ == "__main__":
    list_devices()
    
    # Try testing the default device first
    print("\nTesting System Default Device:")
    test_microphone(None)
    
    # Check if user had a specific problematic index (9) and test it if it exists
    try:
        devs = sd.query_devices()
        if len(devs) > 9 and devs[9]['max_input_channels'] > 0:
            print("\nTesting previously hardcoded Device 9:")
            test_microphone(9)
        elif len(devs) > 9:
             print("\nDevice 9 exists but has 0 input channels.")
        else:
            print("\nDevice 9 does not exist.")
    except Exception:
        pass
