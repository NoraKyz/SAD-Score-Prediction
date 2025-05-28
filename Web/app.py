from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Tải mô hình mà không biên dịch, sau đó biên dịch lại
model = tf.keras.models.load_model(
    'student_score_predictor_cnn.h5', compile=False)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Mô hình đã được tải và biên dịch lại thành công!")

# Định nghĩa trọng số cho từng cột điểm C1 đến C4
INPUT_FEATURE_WEIGHTS = np.array([0.1, 0.1, 0.1, 0.2])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Mô hình chưa được tải hoặc có lỗi khi tải.'}), 500

    try:
        data = request.get_json(force=True)
        scores_str = [data['c1'], data['c2'], data['c3'], data['c4']]

        try:
            scores_actual = np.array([float(s)
                                      for s in scores_str]).reshape(1, -1)
        except ValueError:
            return jsonify({'error': 'Vui lòng nhập giá trị số hợp lệ cho các đầu điểm.'}), 400

        if not np.all((scores_actual >= 0) & (scores_actual <= 10)):
            return jsonify({'error': 'Điểm số phải nằm trong khoảng từ 0 đến 10.'}), 400
        
        if np.any(scores_actual < 1):
            return jsonify({'predicted_final_exam': 0.0})

        scores_weighted = scores_actual * INPUT_FEATURE_WEIGHTS

        num_features = scores_weighted.shape[1]
        scores_cnn_input = scores_weighted.reshape((1, num_features, 1))

        prediction = model.predict(scores_cnn_input)
        predicted_score = float(prediction[0][0])

        predicted_score_rounded = round(np.clip(predicted_score, 0, 10), 2)

        return jsonify({'predicted_final_exam': predicted_score_rounded})

    except KeyError:
        return jsonify({'error': 'Cần nhập đầy đủ cả 4 đầu điểm'}), 400
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        return jsonify({'error': f'Đã xảy ra lỗi: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
