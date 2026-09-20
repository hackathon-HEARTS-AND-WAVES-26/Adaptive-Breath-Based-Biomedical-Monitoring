import time
import random
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for browser dashboard access

# Global state to track current active health profile
current_profile = "healthy"

def generate_sensor_reading(profile="healthy"):
    """
    Simulates hardware register feeds from 8 biomedical modules:
      1. MPX5010DP      -> PEF (L/min)
      2. MAX30102       -> SpO2 (%) & Optical Pulse Rate (BPM)
      3. AD8232 / Hand  -> Lead-I ECG (BPM & Waveform Rhythm Status)
      4. NTC Thermistor -> Lip-Contact Core Temperature (°C)
      5. Grove GSR      -> Galvanic Skin Response (ADC)
      6. MAX9814        -> Acoustic Stethoscope (Wheeze flag 0/1)
      7. SGP30          -> Breath Acetone (PPM) & eCO2 (PPM) over I2C
      8. MQ-138         -> Secondary Ketone/Vapor Array (ADC)
    """
    if profile == "healthy":
        payload = {
            "pressure_pef_lmin": round(random.gauss(460, 12), 1),      # Peak Expiratory Flow
            "spo2_percent": int(random.gauss(98, 0.8)),                 # Blood Oxygen
            "optical_hr_bpm": int(random.gauss(72, 3)),                 # MAX30102 Optical HR
            "ecg_lead1_bpm": int(random.gauss(72, 2)),                  # Hand-Contact Lead-I ECG
            "ecg_lead1_status": "Normal Sinus Rhythm",                  # ECG Waveform Diagnosis
            "lip_temp_celsius": round(random.gauss(36.6, 0.15), 2),     # NTC Lip Contact Temp
            "gsr_adc_value": int(random.gauss(210, 12)),                # Skin Conductance / Stress
            "acoustic_wheeze_detected": 0,                              # MAX9814 Audio Stethoscope
            "sgp30_acetone_ppm": round(random.gauss(0.75, 0.05), 2),   # SGP30 Acetone
            "sgp30_eco2_ppm": int(random.gauss(415, 10)),               # SGP30 eCO2
            "mq138_vapor_adc": int(random.gauss(145, 8))                # MQ-138 Ketone Vapors
        }
    elif profile == "asthma_attack":
        payload = {
            "pressure_pef_lmin": round(random.gauss(205, 18), 1),      # Severe Flow Drop (< 300)
            "spo2_percent": int(random.gauss(90, 2)),                   # SpO2 Drop
            "optical_hr_bpm": int(random.gauss(110, 5)),                # Tachycardia
            "ecg_lead1_bpm": int(random.gauss(112, 4)),                 # ECG Sinus Tachycardia
            "ecg_lead1_status": "Sinus Tachycardia (Stress Response)",
            "lip_temp_celsius": round(random.gauss(36.9, 0.2), 2),
            "gsr_adc_value": int(random.gauss(490, 20)),                # High Sweat / Stress
            "acoustic_wheeze_detected": 1,                              # Wheeze Audio Active!
            "sgp30_acetone_ppm": round(random.gauss(0.95, 0.08), 2),
            "sgp30_eco2_ppm": int(random.gauss(880, 25)),               # Breath eCO2 retention
            "mq138_vapor_adc": int(random.gauss(175, 12))
        }
    elif profile == "dka_alert":
        payload = {
            "pressure_pef_lmin": round(random.gauss(430, 15), 1),
            "spo2_percent": int(random.gauss(95, 1)),
            "optical_hr_bpm": int(random.gauss(98, 4)),
            "ecg_lead1_bpm": int(random.gauss(96, 3)),
            "ecg_lead1_status": "Sinus Rhythm with Mild Elevation",
            "lip_temp_celsius": round(random.gauss(38.7, 0.25), 2),     # High Fever (> 38°C)
            "gsr_adc_value": int(random.gauss(540, 25)),                # High GSR Stress Response
            "acoustic_wheeze_detected": 0,
            "sgp30_acetone_ppm": round(random.gauss(4.35, 0.25), 2),   # High Breath Acetone (> 3.5 PPM)
            "sgp30_eco2_ppm": int(random.gauss(520, 18)),
            "mq138_vapor_adc": int(random.gauss(740, 30))               # High Organic Ketone Vapors
        }
    elif profile == "cardiac_arrhythmia":
        payload = {
            "pressure_pef_lmin": round(random.gauss(390, 20), 1),
            "spo2_percent": int(random.gauss(92, 2)),
            "optical_hr_bpm": int(random.gauss(138, 12)),               # Irregular Pulse
            "ecg_lead1_bpm": int(random.gauss(145, 15)),                # Severe Cardiac Spike
            "ecg_lead1_status": "Arrhythmia / Ventricular Instability", # Critical ECG Anomaly
            "lip_temp_celsius": round(random.gauss(36.5, 0.2), 2),
            "gsr_adc_value": int(random.gauss(610, 35)),                # Acute Sympathetic Panic Spike
            "acoustic_wheeze_detected": 0,
            "sgp30_acetone_ppm": round(random.gauss(0.82, 0.06), 2),
            "sgp30_eco2_ppm": int(random.gauss(440, 15)),
            "mq138_vapor_adc": int(random.gauss(160, 10))
        }
    return payload

@app.route('/api/sensor', methods=['GET'])
def get_sensor_data():
    """Returns real-time telemetry snapshot from simulated hardware registers."""
    data = generate_sensor_reading(profile=current_profile)
    return jsonify({
        "status": "success",
        "active_profile": current_profile,
        "timestamp": time.strftime("%H:%M:%S"),
        "sensor_data": data
    })

@app.route('/api/set_profile', methods=['POST'])
def set_profile():
    """Allows web dashboard to trigger simulation scenario switches."""
    global current_profile
    req_data = request.get_json()
    if req_data and "profile" in req_data:
        current_profile = req_data["profile"]
        print(f"[Python Telemetry Engine]: Switched active health profile to -> {current_profile}")
        return jsonify({"status": "success", "profile": current_profile})
    return jsonify({"status": "error", "message": "Invalid request body"}), 400

if __name__ == '__main__':
    print("\n=======================================================")
    print("=== HEARTZ AND WAVES Telemetry API Running on http://127.0.0.1:5000 ===")
    print("=======================================================\n")
    app.run(host='127.0.0.1', port=5000, debug=True)