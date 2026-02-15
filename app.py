from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import base64
import io
from datetime import datetime
import os
import sys

app = Flask(__name__)
CORS(app)

# Hugging Face API configuration
HF_API_URL = "https://api-inference.huggingface.co/models/Kaludi/food-category-classification-v2.0"
HF_API_TOKEN = os.environ.get('HF_API_TOKEN', '')  # Set this in your environment

print(f"[STARTUP] HF_API_TOKEN is set: {bool(HF_API_TOKEN)}", file=sys.stderr)
print(f"[STARTUP] HF_API_URL: {HF_API_URL}", file=sys.stderr)

# Food database with prices (MKD - Macedonian Denar)
FOOD_DATABASE = {
    "pizza": {"price": 250, "calories": 266, "protein": 11, "carbs": 33, "fat": 10},
    "burger": {"price": 180, "calories": 295, "protein": 17, "carbs": 24, "fat": 14},
    "sandwich": {"price": 120, "calories": 304, "protein": 13, "carbs": 40, "fat": 11},
    "salad": {"price": 150, "calories": 33, "protein": 3, "carbs": 7, "fat": 0.2},
    "pasta": {"price": 200, "calories": 158, "protein": 6, "carbs": 31, "fat": 1},
    "rice": {"price": 80, "calories": 130, "protein": 2.7, "carbs": 28, "fat": 0.3},
    "chicken": {"price": 220, "calories": 239, "protein": 27, "carbs": 0, "fat": 14},
    "beef": {"price": 280, "calories": 250, "protein": 26, "carbs": 0, "fat": 15},
    "fish": {"price": 300, "calories": 206, "protein": 22, "carbs": 0, "fat": 12},
    "soup": {"price": 100, "calories": 71, "protein": 4, "carbs": 9, "fat": 2},
    "french fries": {"price": 90, "calories": 312, "protein": 3.4, "carbs": 41, "fat": 15},
    "bread": {"price": 30, "calories": 265, "protein": 9, "carbs": 49, "fat": 3.2},
    "cheese": {"price": 60, "calories": 402, "protein": 25, "carbs": 1.3, "fat": 33},
    "tomato": {"price": 40, "calories": 18, "protein": 0.9, "carbs": 3.9, "fat": 0.2},
    "potato": {"price": 50, "calories": 77, "protein": 2, "carbs": 17, "fat": 0.1},
    "egg": {"price": 20, "calories": 155, "protein": 13, "carbs": 1.1, "fat": 11},
    "apple": {"price": 35, "calories": 52, "protein": 0.3, "carbs": 14, "fat": 0.2},
    "banana": {"price": 30, "calories": 89, "protein": 1.1, "carbs": 23, "fat": 0.3},
    "orange": {"price": 40, "calories": 47, "protein": 0.9, "carbs": 12, "fat": 0.1},
    "yogurt": {"price": 45, "calories": 59, "protein": 10, "carbs": 3.6, "fat": 0.4},
    "milk": {"price": 55, "calories": 42, "protein": 3.4, "carbs": 5, "fat": 1},
    "coffee": {"price": 70, "calories": 2, "protein": 0.3, "carbs": 0, "fat": 0},
    "tea": {"price": 40, "calories": 1, "protein": 0, "carbs": 0.3, "fat": 0},
    "juice": {"price": 60, "calories": 45, "protein": 0.5, "carbs": 11, "fat": 0.1},
    "water": {"price": 25, "calories": 0, "protein": 0, "carbs": 0, "fat": 0},
    "cake": {"price": 130, "calories": 257, "protein": 3, "carbs": 36, "fat": 12},
    "cookie": {"price": 50, "calories": 502, "protein": 5.9, "carbs": 64, "fat": 24},
    "ice cream": {"price": 80, "calories": 207, "protein": 3.5, "carbs": 24, "fat": 11},
    "chocolate": {"price": 70, "calories": 546, "protein": 4.9, "carbs": 61, "fat": 31},
    "sushi": {"price": 350, "calories": 143, "protein": 6, "carbs": 21, "fat": 3.5},
    "taco": {"price": 160, "calories": 226, "protein": 9, "carbs": 20, "fat": 13},
    "hot dog": {"price": 110, "calories": 290, "protein": 10, "carbs": 23, "fat": 18},
    "bacon": {"price": 95, "calories": 541, "protein": 37, "carbs": 1.4, "fat": 42},
    "sausage": {"price": 140, "calories": 301, "protein": 12, "carbs": 1.5, "fat": 27},
    "mushroom": {"price": 85, "calories": 22, "protein": 3.1, "carbs": 3.3, "fat": 0.3},
    "onion": {"price": 35, "calories": 40, "protein": 1.1, "carbs": 9, "fat": 0.1},
    "pepper": {"price": 55, "calories": 20, "protein": 0.9, "carbs": 4.6, "fat": 0.2},
    "cucumber": {"price": 45, "calories": 16, "protein": 0.7, "carbs": 3.6, "fat": 0.1},
    "carrot": {"price": 40, "calories": 41, "protein": 0.9, "carbs": 10, "fat": 0.2},
    "broccoli": {"price": 70, "calories": 34, "protein": 2.8, "carbs": 7, "fat": 0.4},
}

def query_huggingface(image_bytes):
    """Query Hugging Face API for food recognition"""
    print(f"[HF] Starting query to Hugging Face", file=sys.stderr)
    print(f"[HF] Image size: {len(image_bytes)} bytes", file=sys.stderr)
    print(f"[HF] Token present: {bool(HF_API_TOKEN)}", file=sys.stderr)
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    try:
        print(f"[HF] Sending POST request to {HF_API_URL}", file=sys.stderr)
        response = requests.post(HF_API_URL, headers=headers, data=image_bytes, timeout=30)
        print(f"[HF] Response status: {response.status_code}", file=sys.stderr)
        print(f"[HF] Response headers: {dict(response.headers)}", file=sys.stderr)
        
        if response.status_code != 200:
            print(f"[HF] ERROR Response text: {response.text}", file=sys.stderr)
        
        response.raise_for_status()
        result = response.json()
        print(f"[HF] Success! Got {len(result)} predictions", file=sys.stderr)
        return result
    except requests.exceptions.Timeout as e:
        print(f"[HF] ERROR Timeout: {e}", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"[HF] ERROR Request exception: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            print(f"[HF] ERROR Response text: {e.response.text}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[HF] ERROR Unexpected: {e}", file=sys.stderr)
        return None

def match_food_items(predictions):
    """Match predictions to food database"""
    print(f"[MATCH] Matching predictions to database", file=sys.stderr)
    matched_items = []
    seen_foods = set()
    
    for pred in predictions[:10]:  # Check top 10 predictions
        label = pred.get('label', '').lower()
        score = pred.get('score', 0)
        print(f"[MATCH] Checking: {label} (score: {score})", file=sys.stderr)
        
        # Try to match with database
        for food_key in FOOD_DATABASE.keys():
            if food_key in label and food_key not in seen_foods and score > 0.05:
                food_info = FOOD_DATABASE[food_key].copy()
                food_info['name'] = food_key.capitalize()
                food_info['confidence'] = round(score * 100, 2)
                matched_items.append(food_info)
                seen_foods.add(food_key)
                print(f"[MATCH] MATCHED: {food_key}", file=sys.stderr)
                break
    
    # If no matches found, return most common canteen items
    if not matched_items:
        print(f"[MATCH] No matches found, using defaults", file=sys.stderr)
        default_items = ['rice', 'chicken', 'salad']
        for item in default_items:
            food_info = FOOD_DATABASE[item].copy()
            food_info['name'] = item.capitalize()
            food_info['confidence'] = 85.0
            matched_items.append(food_info)
    
    print(f"[MATCH] Total matched items: {len(matched_items)}", file=sys.stderr)
    return matched_items

def calculate_totals(items):
    """Calculate total price and nutrition"""
    total_price = sum(item['price'] for item in items)
    total_calories = sum(item['calories'] for item in items)
    total_protein = sum(item['protein'] for item in items)
    total_carbs = sum(item['carbs'] for item in items)
    total_fat = sum(item['fat'] for item in items)
    
    return {
        'total_price': total_price,
        'total_calories': round(total_calories, 1),
        'total_protein': round(total_protein, 1),
        'total_carbs': round(total_carbs, 1),
        'total_fat': round(total_fat, 1)
    }

def generate_receipt_html(items, totals, timestamp):
    """Generate HTML receipt"""
    items_html = ""
    for item in items:
        items_html += f"""
        <tr>
            <td>{item['name']}</td>
            <td>{item['price']} MKD</td>
            <td>{item['calories']} kcal</td>
            <td>{item['protein']}g</td>
            <td>{item['carbs']}g</td>
            <td>{item['fat']}g</td>
            <td>{item['confidence']}%</td>
        </tr>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Courier New', monospace;
                max-width: 800px;
                margin: 40px auto;
                padding: 20px;
                background: #f5f5f5;
            }}
            .receipt {{
                background: white;
                padding: 30px;
                border: 2px dashed #333;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                border-bottom: 2px solid #333;
                padding-bottom: 20px;
                margin-bottom: 20px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 28px;
            }}
            .header p {{
                margin: 5px 0;
                color: #666;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background: #f0f0f0;
                font-weight: bold;
            }}
            .totals {{
                border-top: 3px double #333;
                padding-top: 15px;
                margin-top: 20px;
            }}
            .totals table {{
                margin: 0;
            }}
            .totals td {{
                border: none;
                padding: 5px 10px;
            }}
            .total-row {{
                font-weight: bold;
                font-size: 18px;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding-top: 20px;
                border-top: 2px solid #333;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="receipt">
            <div class="header">
                <h1>🍽️ CAMPUS CANTEEN</h1>
                <p>AI-Powered Food Recognition System</p>
                <p>Receipt Date: {timestamp}</p>
            </div>
            
            <h2>Detected Items</h2>
            <table>
                <thead>
                    <tr>
                        <th>Item</th>
                        <th>Price</th>
                        <th>Calories</th>
                        <th>Protein</th>
                        <th>Carbs</th>
                        <th>Fat</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody>
                    {items_html}
                </tbody>
            </table>
            
            <div class="totals">
                <h2>Totals</h2>
                <table>
                    <tr>
                        <td>Total Price:</td>
                        <td class="total-row">{totals['total_price']} MKD</td>
                    </tr>
                    <tr>
                        <td>Total Calories:</td>
                        <td class="total-row">{totals['total_calories']} kcal</td>
                    </tr>
                    <tr>
                        <td>Total Protein:</td>
                        <td>{totals['total_protein']}g</td>
                    </tr>
                    <tr>
                        <td>Total Carbohydrates:</td>
                        <td>{totals['total_carbs']}g</td>
                    </tr>
                    <tr>
                        <td>Total Fat:</td>
                        <td>{totals['total_fat']}g</td>
                    </tr>
                </table>
            </div>
            
            <div class="footer">
                <p>Thank you for using our AI Food Recognition System!</p>
                <p>Mobile Wireless Networks Project</p>
                <p>Powered by Hugging Face AI</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "food-recognition-api"})

@app.route('/analyze', methods=['POST'])
def analyze_food():
    """Analyze food image and return results"""
    print(f"\n[ANALYZE] ========== NEW REQUEST ==========", file=sys.stderr)
    try:
        if 'image' not in request.files:
            print(f"[ANALYZE] ERROR: No image in request", file=sys.stderr)
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        print(f"[ANALYZE] Image filename: {image_file.filename}", file=sys.stderr)
        image_bytes = image_file.read()
        print(f"[ANALYZE] Image size: {len(image_bytes)} bytes", file=sys.stderr)
        
        # Query Hugging Face API
        print(f"[ANALYZE] Calling Hugging Face API...", file=sys.stderr)
        predictions = query_huggingface(image_bytes)
        
        if not predictions:
            print(f"[ANALYZE] ERROR: No predictions returned from HF", file=sys.stderr)
            return jsonify({
                "error": "Failed to analyze image", 
                "details": "Hugging Face API returned no predictions. Check backend logs."
            }), 500
        
        print(f"[ANALYZE] Got {len(predictions)} predictions from HF", file=sys.stderr)
        
        # Match predictions to food database
        matched_items = match_food_items(predictions)
        print(f"[ANALYZE] Matched {len(matched_items)} items", file=sys.stderr)
        
        # Calculate totals
        totals = calculate_totals(matched_items)
        print(f"[ANALYZE] Calculated totals: {totals}", file=sys.stderr)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate receipt HTML
        receipt_html = generate_receipt_html(matched_items, totals, timestamp)
        
        print(f"[ANALYZE] SUCCESS! Returning results", file=sys.stderr)
        return jsonify({
            "success": True,
            "items": matched_items,
            "totals": totals,
            "timestamp": timestamp,
            "receipt_html": receipt_html
        })
    
    except Exception as e:
        print(f"[ANALYZE] ERROR Exception: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return jsonify({
            "error": "Failed to analyze image",
            "details": str(e),
            "type": type(e).__name__
        }), 500

@app.route('/download-receipt', methods=['POST'])
def download_receipt():
    """Generate and return receipt as downloadable HTML file"""
    try:
        data = request.get_json()
        receipt_html = data.get('receipt_html', '')
        
        if not receipt_html:
            return jsonify({"error": "No receipt data provided"}), 400
        
        # Create file-like object
        receipt_bytes = io.BytesIO(receipt_html.encode('utf-8'))
        receipt_bytes.seek(0)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"receipt_{timestamp}.html"
        
        return send_file(
            receipt_bytes,
            mimetype='text/html',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        print(f"Error in download_receipt: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
