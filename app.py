from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_file,
    flash,
    redirect,
    url_for,
    session,
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
import pickle
import numpy as np
import json
import os
import atexit
from datetime import datetime, timedelta
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from icrawler.builtin import BingImageCrawler
from playwright.sync_api import sync_playwright
import base64
import random
import sklearn  # noqa: F401
import joblib
from flask import render_template_string
import threading
import time

import shutil
import io


active_sessions = {}


# Update your app configuration
app = Flask(__name__)
app.secret_key = "your-secret-key-change-this-to-something-secure"  # Change this to a secure secret key
app.permanent_session_lifetime = 3600

# Configuration
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"csv", "txt", "pkl"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/images/diet_images", exist_ok=True)
os.makedirs("diet_pdfs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Global variable to track crawling status
crawling_status = {}


def crawl_food_image(food_name, max_retries=3):
    """Crawl a single food image using Bing image crawler"""
    try:
        # Clean the food name for filename
        safe_name = "".join(
            c for c in food_name.lower().replace(" ", "_") if c.isalnum() or c in "_-"
        )
        output_dir = f"static/images/diet_images/{safe_name}"
        image_path = f"{output_dir}/000001.jpg"

        # Check if image already exists
        if os.path.exists(image_path):
            return f"/static/images/diet_images/{safe_name}/000001.jpg"

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set up Bing crawler with retries
        for attempt in range(max_retries):
            try:
                crawler = BingImageCrawler(
                    storage={"root_dir": output_dir},
                    log_level=30,  # WARNING level to reduce verbose output
                    downloader_threads=10,
                    parser_threads=10,
                    feeder_threads=10,
                )

                # Search for food images with more specific query
                search_query = f"{food_name} food dish recipe"
                print(f"Searching Bing for: {search_query}")

                crawler.crawl(keyword=search_query, max_num=1, file_idx_offset=0)

                # Check if image was downloaded
                if os.path.exists(image_path):
                    print(f"Successfully downloaded image for: {food_name}")
                    return f"/static/images/diet_images/{safe_name}/000001.jpg"
                else:
                    print(f"Attempt {attempt + 1}: No image found for {food_name}")

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {food_name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait longer before retry for Bing

        # If all attempts failed, return placeholder
        print(f"All attempts failed for {food_name}, using placeholder")
        return "/static/images/placeholder.jpg"

    except Exception as e:
        print(f"Error crawling image for {food_name}: {e}")
        return "/static/images/placeholder.jpg"


def crawl_diet_images_async(diet_charts):
    """Asynchronously crawl images for all food items in diet charts"""
    global crawling_status
    try:
        print("Starting asynchronous Bing image crawling...")
        food_items = set()

        # Collect all unique food items
        for chart in diet_charts:
            for meal_type in ["breakfast", "lunch", "snacks", "dinner"]:
                if meal_type in chart and chart[meal_type]:
                    for item in chart[meal_type]:
                        if "name" in item:
                            food_items.add(item["name"])

        print(f"Found {len(food_items)} unique food items to crawl from Bing")

        # Crawl images for each food item
        for i, food_name in enumerate(food_items):
            if food_name not in crawling_status:
                crawling_status[food_name] = "crawling"
                try:
                    print(f"Crawling image {i + 1}/{len(food_items)}: {food_name}")
                    image_path = crawl_food_image(food_name)
                    crawling_status[food_name] = image_path
                    print(f"Crawled image for: {food_name} -> {image_path}")

                    # Small delay between requests to be respectful to Bing
                    time.sleep(1)

                except Exception as e:
                    print(f"Failed to crawl image for {food_name}: {e}")
                    crawling_status[food_name] = "/static/images/placeholder.jpg"

        print("Bing image crawling completed")

    except Exception as e:
        print(f"Error in async Bing image crawling: {e}")


def crawl_food_image_batch(food_name, max_retries=2):
    """Optimized function to crawl a single food image with better error handling"""
    try:
        # Clean the food name for filename
        safe_name = "".join(
            c for c in food_name.lower().replace(" ", "_") if c.isalnum() or c in "_-"
        )
        output_dir = f"static/images/diet_images/{safe_name}"
        image_path = f"{output_dir}/000001.jpg"

        # Check if image already exists
        if os.path.exists(image_path):
            return f"/static/images/diet_images/{safe_name}/000001.jpg"

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set up Bing crawler with optimized settings for batch processing
        for attempt in range(max_retries):
            try:
                crawler = BingImageCrawler(
                    storage={"root_dir": output_dir},
                    log_level=40,  # ERROR level only to reduce output
                    downloader_threads=20,  # Increased threads
                    parser_threads=20,  # Increased threads
                    feeder_threads=20,  # Increased threads
                )

                # Search for food images with specific query
                search_query = f"{food_name} food dish"

                crawler.crawl(keyword=search_query, max_num=1, file_idx_offset=0)

                # Check if image was downloaded
                if os.path.exists(image_path):
                    print(f"✓ Downloaded image for: {food_name}")
                    return f"/static/images/diet_images/{safe_name}/000001.jpg"
                else:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)  # Shorter wait between retries

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {food_name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)

        # If all attempts failed, return placeholder
        print(f"All attempts failed for {food_name}, using placeholder")
        return "/static/images/placeholder.jpg"

    except Exception as e:
        print(f"Error crawling image for {food_name}: {e}")
        return "/static/images/placeholder.jpg"


def crawl_diet_images_async_optimized(diet_charts):
    """Asynchronously crawl images for all food items using concurrent threading"""
    global crawling_status
    try:
        print("Starting optimized concurrent Bing image crawling...")
        food_items = set()

        # Collect all unique food items
        for chart in diet_charts:
            for meal_type in ["breakfast", "lunch", "snacks", "dinner"]:
                if meal_type in chart and chart[meal_type]:
                    for item in chart[meal_type]:
                        if "name" in item:
                            food_items.add(item["name"])

        food_list = list(food_items)
        print(f"Found {len(food_list)} unique food items to crawl")

        # Mark all as crawling initially
        for food_name in food_list:
            if food_name not in crawling_status:
                crawling_status[food_name] = "crawling"

        # Use ThreadPoolExecutor for concurrent downloads
        # Limit to 10 concurrent downloads to be respectful to Bing
        max_workers = min(10, len(food_list))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all crawling tasks
            future_to_food = {
                executor.submit(crawl_food_image_batch, food_name): food_name
                for food_name in food_list
            }

            # Process completed downloads
            completed = 0
            for future in as_completed(future_to_food):
                food_name = future_to_food[future]
                try:
                    image_path = future.result()
                    crawling_status[food_name] = image_path
                    completed += 1
                    print(f"✓ Completed ({completed}/{len(food_list)}): {food_name}")

                except Exception as e:
                    print(f"✗ Failed to crawl {food_name}: {e}")
                    crawling_status[food_name] = "/static/images/placeholder.jpg"
                    completed += 1

        print(f"✅ Concurrent image crawling completed! Downloaded {completed} images")

    except Exception as e:
        print(f"Error in optimized async image crawling: {e}")


def start_image_crawling_for_diet_optimized(diet_charts):
    """Start optimized concurrent image crawling when diet is generated"""
    print("Starting immediate concurrent image crawling for generated diet...")
    threading.Thread(
        target=crawl_diet_images_async_optimized, args=(diet_charts,), daemon=True
    ).start()


# Alternative approach: Batch crawling with icrawler for multiple keywords at once
def crawl_multiple_foods_single_session(food_list, batch_size=10):
    """Alternative: Use single crawler session for multiple foods"""
    global crawling_status
    try:
        print(f"Starting batch crawling for {len(food_list)} foods...")

        # Process foods in batches
        for i in range(0, len(food_list), batch_size):
            batch = food_list[i : i + batch_size]
            print(f"Processing batch {i // batch_size + 1}: {batch}")

            # Mark batch as crawling
            for food_name in batch:
                crawling_status[food_name] = "crawling"

            # Use single crawler instance for the batch
            base_dir = "static/images/diet_images"
            os.makedirs(base_dir, exist_ok=True)

            crawler = BingImageCrawler(
                storage={"root_dir": base_dir},
                log_level=40,
                downloader_threads=30,
                parser_threads=30,
                feeder_threads=30,
            )

            # Crawl each food in the batch
            for food_name in batch:
                try:
                    safe_name = "".join(
                        c
                        for c in food_name.lower().replace(" ", "_")
                        if c.isalnum() or c in "_-"
                    )

                    # Check if already exists
                    expected_path = f"{base_dir}/{safe_name}/000001.jpg"
                    if os.path.exists(expected_path):
                        crawling_status[food_name] = (
                            f"/static/images/diet_images/{safe_name}/000001.jpg"
                        )
                        continue

                    # Create specific subfolder for this food
                    food_dir = f"{base_dir}/{safe_name}"
                    os.makedirs(food_dir, exist_ok=True)

                    # Update crawler storage for this food
                    crawler.storage.root_dir = food_dir

                    search_query = f"{food_name} food dish recipe"
                    crawler.crawl(keyword=search_query, max_num=1)

                    # Check if downloaded successfully
                    if os.path.exists(f"{food_dir}/000001.jpg"):
                        crawling_status[food_name] = (
                            f"/static/images/diet_images/{safe_name}/000001.jpg"
                        )
                        print(f"✓ Batch downloaded: {food_name}")
                    else:
                        crawling_status[food_name] = "/static/images/placeholder.jpg"
                        print(f"✗ Batch failed: {food_name}")

                except Exception as e:
                    print(f"Error in batch crawling {food_name}: {e}")
                    crawling_status[food_name] = "/static/images/placeholder.jpg"

            # Small delay between batches
            if i + batch_size < len(food_list):
                time.sleep(1)

        print("✅ Batch crawling completed!")

    except Exception as e:
        print(f"Error in batch crawling: {e}")


def start_batch_crawling_for_diet(diet_charts):
    """Start batch crawling approach"""

    def extract_and_crawl():
        food_items = set()
        for chart in diet_charts:
            for meal_type in ["breakfast", "lunch", "snacks", "dinner"]:
                if meal_type in chart and chart[meal_type]:
                    for item in chart[meal_type]:
                        if "name" in item:
                            food_items.add(item["name"])

        crawl_multiple_foods_single_session(list(food_items), batch_size=10)

    threading.Thread(target=extract_and_crawl, daemon=True).start()


# Add this route to your Flask app.py file


@app.route("/model-statistics")
def model_statistics():
    """
    Display comprehensive model statistics and performance metrics
    """
    return render_template("model_statistics.html")


# Add these functions to your existing app.py file


@app.route("/generate-diet", methods=["POST"])
def generate_diet():
    """Generate personalized diet charts based on risk assessment with hybrid support"""
    try:
        # Get user preferences
        diet_type = request.form.get("diet_type", "vegetarian")
        num_charts = int(request.form.get("num_charts", 3))
        veg_percentage = (
            int(request.form.get("veg_percentage", 50))
            if diet_type == "hybrid"
            else None
        )

        # Validate number of charts
        if num_charts < 1 or num_charts > 10:
            num_charts = 3

        # Load outcome from session first, then pickle as fallback
        outcome = None
        if "prediction_result" in session:
            outcome = session["prediction_result"]
        else:
            with open("static/diet_images/outcome.pkl", "rb") as f:
                outcome = pickle.load(f)

        if not diet_plan:
            flash(
                "Diet plan data not available. Please check data/Final_diet_plan.json file.",
                "error",
            )
            return redirect(url_for("diet_plan_page"))

        # Get diet recommendation based on risk category
        risk_category = outcome["risk_category"]

        print(f"Generating diet for: {diet_type}, {risk_category}")
        if diet_type == "hybrid":
            print(f"Hybrid diet with {veg_percentage}% vegetarian meals")

        try:
            # Generate diet charts based on diet type
            if diet_type == "hybrid":
                diet_charts = create_hybrid_diet_charts(
                    diet_plan, risk_category, veg_percentage, num_charts
                )
            else:
                # Access diet data from JSON structure for pure veg/non-veg
                diet_data = diet_plan["diet_plan"][diet_type][risk_category]
                print(f"Found diet data with meals: {list(diet_data.keys())}")
                diet_charts = create_random_diet_charts(diet_data, num_charts)

        except KeyError as e:
            print(f"KeyError accessing diet data: {e}")
            available_categories = list(
                diet_plan.get("diet_plan", {})
                .get(diet_type if diet_type != "hybrid" else "vegetarian", {})
                .keys()
            )
            flash(
                f"No diet plan found for {risk_category}. Available categories: {available_categories}",
                "error",
            )
            return redirect(url_for("diet_plan_page"))

        if not diet_charts:
            flash("Could not generate diet charts. Please check diet data.", "error")
            return redirect(url_for("diet_plan_page"))

        # Find recommended chart (one closest to average calories)
        if len(diet_charts) > 1:
            total_calories = [chart.get("total_calories", 0) for chart in diet_charts]
            avg_calories = sum(total_calories) / len(total_calories)
            closest_chart = min(
                diet_charts,
                key=lambda x: abs(x.get("total_calories", 0) - avg_calories),
            )
            recommended_chart_num = closest_chart["chart_number"]
        else:
            recommended_chart_num = diet_charts[0]["chart_number"]

        # Mark recommended chart
        for chart in diet_charts:
            chart["is_recommended"] = chart["chart_number"] == recommended_chart_num

        print(
            f"Generated {len(diet_charts)} diet charts, recommended: Chart {recommended_chart_num}"
        )

        # Store diet charts and preferences in session for PDF generation
        session["diet_charts"] = diet_charts
        session["diet_type"] = diet_type
        session["veg_percentage"] = veg_percentage
        session["is_vegetarian"] = (
            diet_type == "vegetarian"
        )  # Keep for backward compatibility
        session.permanent = True

        # Start crawling images IMMEDIATELY when diet is generated
        start_image_crawling_for_diet_optimized(diet_charts)

        return render_template(
            "diet_charts.html",
            diet_charts=diet_charts,
            outcome=outcome,
            diet_type=diet_type,
            veg_percentage=veg_percentage,
            is_vegetarian=diet_type == "vegetarian",
        )

    except FileNotFoundError:
        flash("Please complete heart disease prediction first.", "warning")
        return redirect(url_for("predict"))
    except Exception as e:
        print(f"Error in generate_diet: {e}")
        flash(f"Error generating diet plan: {str(e)}", "error")
        return redirect(url_for("diet_plan_page"))


def create_hybrid_diet_charts(
    diet_plan_data, risk_category, veg_percentage, num_charts=3
):
    """Create hybrid diet charts mixing vegetarian and non-vegetarian meals"""
    if not diet_plan_data or "diet_plan" not in diet_plan_data:
        print("No diet plan data provided")
        return []

    try:
        # Get both vegetarian and non-vegetarian data
        veg_data = diet_plan_data["diet_plan"]["vegetarian"][risk_category]
        non_veg_data = diet_plan_data["diet_plan"]["non_vegetarian"][risk_category]

        print(
            f"Creating {num_charts} hybrid diet charts with {veg_percentage}% vegetarian meals"
        )
        print(f"Veg meals available: {list(veg_data.keys())}")
        print(f"Non-veg meals available: {list(non_veg_data.keys())}")

    except KeyError as e:
        print(f"Error accessing diet data for hybrid: {e}")
        return []

    # Define meal configuration
    meal_config = {"breakfast": 2, "lunch": 3, "snacks": 2, "dinner": 3}

    diet_charts = []

    for chart_num in range(1, num_charts + 1):
        chart = {
            "chart_number": chart_num,
            "total_calories": 0,
            "diet_type": "hybrid",
            "veg_percentage": veg_percentage,
        }

        for meal_type, num_items in meal_config.items():
            # Get available items from both diet types
            veg_items = veg_data.get(meal_type, [])
            non_veg_items = non_veg_data.get(meal_type, [])

            if not veg_items and not non_veg_items:
                chart[meal_type] = []
                continue

            # Calculate how many vegetarian vs non-vegetarian items to include
            veg_count = max(1, round(num_items * veg_percentage / 100))
            non_veg_count = num_items - veg_count

            # Adjust if we don't have enough items
            available_veg = min(veg_count, len(veg_items))
            available_non_veg = min(non_veg_count, len(non_veg_items))

            # If one category is short, compensate with the other
            if available_veg < veg_count and len(non_veg_items) > available_non_veg:
                available_non_veg = min(
                    available_non_veg + (veg_count - available_veg), len(non_veg_items)
                )
            elif available_non_veg < non_veg_count and len(veg_items) > available_veg:
                available_veg = min(
                    available_veg + (non_veg_count - available_non_veg), len(veg_items)
                )

            selected_items = []

            # Select vegetarian items
            if available_veg > 0 and veg_items:
                selected_veg = random.sample(veg_items, available_veg)
                for item in selected_veg:
                    item_copy = item.copy()
                    item_copy["diet_source"] = "vegetarian"
                    selected_items.append(item_copy)

            # Select non-vegetarian items
            if available_non_veg > 0 and non_veg_items:
                selected_non_veg = random.sample(non_veg_items, available_non_veg)
                for item in selected_non_veg:
                    item_copy = item.copy()
                    item_copy["diet_source"] = "non_vegetarian"
                    selected_items.append(item_copy)

            # Shuffle the combined items for variety
            random.shuffle(selected_items)

            chart[meal_type] = selected_items

            # Calculate calories for this meal
            meal_calories = sum(item.get("calories", 0) for item in selected_items)
            chart["total_calories"] += meal_calories

        diet_charts.append(chart)

    return diet_charts


def create_random_diet_charts(diet_data, num_charts=3):
    """Create random diet charts from available diet data (updated to handle diet_source)"""
    if not diet_data:
        print("No diet data provided")
        return []

    # Define meal configuration (items per meal type)
    meal_config = {"breakfast": 2, "lunch": 3, "snacks": 2, "dinner": 3}

    diet_charts = []

    print(f"Creating {num_charts} diet charts from meals: {list(diet_data.keys())}")

    for chart_num in range(1, num_charts + 1):
        chart = {"chart_number": chart_num, "total_calories": 0}

        for meal_type, num_items in meal_config.items():
            if meal_type in diet_data and diet_data[meal_type]:
                available_items = diet_data[meal_type]

                # Ensure we don't try to sample more items than available
                items_to_select = min(num_items, len(available_items))

                if items_to_select > 0:
                    # Randomly select items for this meal
                    selected_items = random.sample(available_items, items_to_select)

                    # Add diet_source information for consistency
                    for item in selected_items:
                        if "diet_source" not in item:
                            # Infer from the chart type or set as unknown
                            item["diet_source"] = "unknown"

                    chart[meal_type] = selected_items

                    # Calculate calories for this meal
                    meal_calories = sum(
                        item.get("calories", 0) for item in selected_items
                    )
                    chart["total_calories"] += meal_calories
                else:
                    chart[meal_type] = []
            else:
                chart[meal_type] = []
                print(f"Warning: No {meal_type} data available")

        diet_charts.append(chart)

    return diet_charts


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        self.setFillColor(HexColor("#6b7280"))
        self.drawRightString(
            A4[0] - 20*mm, 10*mm,
            f"Page {self._pageNumber} of {page_count}"
        )
        # Add footer text
        self.drawString(
            20*mm, 10*mm,
            "CarePulse - Personalized Health & Diet Plan"
        )


def encode_image_to_base64(image_path):
    """Convert image file to base64 for embedding in PDF"""
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:image/jpeg;base64,{encoded}"
        return None
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


@app.template_filter('b64encode')
def b64encode_filter(image_path):
    """Jinja filter to encode images to base64"""
    return encode_image_to_base64(image_path)


def generate_pdf_with_playwright(html_content):
    """Generate PDF from HTML using Playwright/Chromium"""
    try:
        with sync_playwright() as p:
            # Launch browser in headless mode
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            # Set content and wait for it to load
            page.set_content(html_content, wait_until="networkidle")
            
            # Give extra time for images to render
            page.wait_for_timeout(2000)
            
            # Generate PDF with professional settings
            pdf_bytes = page.pdf(
                format="A4",
                print_background=True,  # Important: enables background colors
                margin={
                    "top": "15mm",
                    "bottom": "15mm",
                    "left": "15mm",
                    "right": "15mm"
                },
                prefer_css_page_size=False,
            )
            
            browser.close()
            
            # Return as BytesIO buffer for Flask send_file
            return io.BytesIO(pdf_bytes)
            
    except Exception as e:
        print(f"Error in Playwright PDF generation: {e}")
        import traceback
        traceback.print_exc()
        raise


# 3. ADD THIS NEW ROUTE (this replaces your old download_pdf route)

@app.route("/download_pdf_v2")
def download_pdf_v2():
    """Generate and download professional PDF using Playwright with images"""
    try:
        # Load outcome from session first
        outcome = session.get("prediction_result")
        if not outcome:
            # Fallback to pickle file
            try:
                with open("static/diet_images/outcome.pkl", "rb") as f:
                    outcome = pickle.load(f)
            except FileNotFoundError:
                flash("Please complete heart disease prediction first.", "warning")
                return redirect(url_for("predict"))

        # Get diet charts from session
        diet_charts = session.get("diet_charts")
        if not diet_charts:
            flash("Please generate diet charts first.", "warning")
            return redirect(url_for("diet_plan_page"))

        # Add diet preferences to outcome
        diet_type = session.get("diet_type", "vegetarian")
        outcome["diet_type"] = diet_type
        outcome["veg_percentage"] = session.get("veg_percentage", None)

        # Render HTML template for PDF
        html_content = render_template(
            "pdf_template.html",
            outcome=outcome,
            diet_charts=diet_charts,
            diet_type=diet_type,
            veg_percentage=outcome.get("veg_percentage"),
            generation_date=datetime.now(),
        )

        # Generate PDF using Playwright
        pdf_buffer = generate_pdf_with_playwright(html_content)

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        diet_type_suffix = f"_{diet_type}"
        if diet_type == "hybrid":
            diet_type_suffix += f"_{outcome['veg_percentage']}pct_veg"
        filename = f"CarePulse_Diet_Plan{diet_type_suffix}_{timestamp}.pdf"

        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name=filename,
            mimetype="application/pdf",
        )

    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        flash(f"Error generating PDF: {str(e)}", "error")
        return redirect(url_for("diet_plan_page"))


# 4. REGISTER THE TEMPLATE FILTER (add after app initialization)
# Find where you have: app = Flask(__name__)
# Add this line after it:
app.jinja_env.filters['b64encode'] = b64encode_filter


@app.route("/model-statistics-detailed")
def model_statistics_detailed():
    """
    Display detailed model statistics with actual computed metrics
    """
    # Model performance metrics (replace with actual values from your model)
    performance_metrics = {
        "accuracy": 0.946,
        "precision": 0.866,
        "recall": 0.927,
        "f1_score": 0.927,
        "roc_auc": 0.927,
    }

    # Confusion matrix values (replace with actual values)
    confusion_matrix = {
        "true_negatives": 1598,
        "false_positives": 2,
        "false_negatives": 143,
        "true_positives": 924,
    }

    # Feature correlations (replace with actual computed correlations)
    feature_correlations = {
        "positive_factors": [
            {"name": "BMI (Body Mass Index)", "correlation": 0.020},
            {"name": "Stress Level", "correlation": 0.016},
            {"name": "High LDL Cholesterol", "correlation": 0.008},
            {"name": "Homocysteine Level", "correlation": 0.008},
            {"name": "Exercise Habits", "correlation": 0.005},
        ],
        "negative_factors": [
            {"name": "Alcohol Consumption", "correlation": -0.018},
            {"name": "Gender", "correlation": -0.017},
            {"name": "Blood Pressure", "correlation": -0.014},
            {"name": "Sugar Consumption", "correlation": -0.013},
            {"name": "Age", "correlation": -0.009},
        ],
    }

    # Model hyperparameters
    model_config = {
        "algorithm": "Random Forest Classifier",
        "n_estimators": 400,
        "max_depth": 30,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": "balanced",
    }

    # Dataset information
    dataset_info = {
        "total_samples": 8763,
        "features": 19,
        "training_samples": 7010,
        "test_samples": 1753,
        "positive_cases": 1067,
        "negative_cases": 7696,
        "positive_percentage": 12.2,
        "negative_percentage": 87.8,
    }

    return render_template(
        "model_statistics.html",
        performance_metrics=performance_metrics,
        confusion_matrix=confusion_matrix,
        feature_correlations=feature_correlations,
        model_config=model_config,
        dataset_info=dataset_info,
    )


# Helper function to save your matplotlib plots as images
def save_model_plots():
    """
    Save all your model analysis plots to the static/images directory
    Call this function after training your model to generate all required images
    """
    import os

    # Ensure the images directory exists
    os.makedirs("static/images", exist_ok=True)

    # You would save your plots here like:
    # plt.figure(figsize=(12, 5))
    # # Your class weight analysis plot code
    # plt.savefig('static/images/class_weight_analysis.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # Repeat for all other plots...
    print("Model plots saved successfully!")


# Add new route to get image for food item
@app.route("/get-food-image/<food_name>")
def get_food_image(food_name):
    """Get image path for a food item"""
    global crawling_status
    try:
        if food_name in crawling_status:
            if crawling_status[food_name] == "crawling":
                return jsonify({"status": "loading", "image_path": None})
            else:
                return jsonify(
                    {"status": "ready", "image_path": crawling_status[food_name]}
                )
        else:
            # Start crawling this image immediately
            def crawl_single():
                crawling_status[food_name] = "crawling"
                image_path = crawl_food_image(food_name)
                crawling_status[food_name] = image_path

            threading.Thread(target=crawl_single).start()
            return jsonify({"status": "loading", "image_path": None})

    except Exception as e:
        print(f"Error getting food image: {e}")
        return jsonify(
            {"status": "error", "image_path": "/static/images/placeholder.jpg"}
        )


# Create placeholder image if it doesn't exist
def create_placeholder_image():
    """Create a placeholder image file"""
    placeholder_path = "static/images/placeholder.jpg"
    if not os.path.exists(placeholder_path):
        os.makedirs("static/images", exist_ok=True)
        # Create a simple placeholder (you can replace this with an actual image file)
        with open(placeholder_path.replace(".jpg", ".txt"), "w") as f:
            f.write("placeholder")


# Load models and data
# Add this improved load_models function to replace your existing one


def load_models():
    """Load all required models and encoders from your trained model"""
    try:
        # Update with your actual model file paths
        model_files = {
            "model": "models/heart_disease_model_20251207_155704.pkl",
            "encoders": "models/label_encoders_20251207_155704.pkl",
            "features": "models/feature_info_20251207_155704.pkl",
            "metadata": "models/model_metadata_20251207_155704.pkl",
        }

        loaded_models = {}

        print("\n" + "=" * 60)
        print("🔄 Loading Models...")
        print("=" * 60)

        for name, path in model_files.items():
            if os.path.exists(path):
                try:
                    # Try joblib first, then pickle
                    if path.endswith(".pkl"):
                        try:
                            loaded_models[name] = joblib.load(path)
                            print(f"✓ Loaded {name} using joblib from {path}")
                        except Exception as joblib_error:
                            print(f"  Joblib failed for {name}: {joblib_error}")
                            try:
                                with open(path, "rb") as f:
                                    loaded_models[name] = pickle.load(f)
                                print(f"✓ Loaded {name} using pickle from {path}")
                            except Exception as pickle_error:
                                print(f"✗ Both joblib and pickle failed for {name}")
                                print(f"  Joblib error: {joblib_error}")
                                print(f"  Pickle error: {pickle_error}")

                except Exception as e:
                    print(f"✗ Error loading {name}: {e}")
            else:
                print(f"✗ File not found: {path}")

        # Verify model is actually loaded
        if "model" in loaded_models and loaded_models["model"] is not None:
            print(f"\n✅ Model successfully loaded!")
            print(f"   Model type: {type(loaded_models['model'])}")

            # Check if model has predict method
            if hasattr(loaded_models["model"], "predict"):
                print(f"   ✓ Model has predict method")
            if hasattr(loaded_models["model"], "predict_proba"):
                print(f"   ✓ Model has predict_proba method")
        else:
            print(f"\n❌ WARNING: Model not loaded successfully!")
            return {}

        print("=" * 60 + "\n")
        return loaded_models

    except Exception as e:
        print(f"❌ Critical error loading models: {e}")
        import traceback

        traceback.print_exc()
        return {}


# Add this debug route to test if model is loaded
@app.route("/debug-model")
def debug_model():
    """Debug endpoint to check model status"""
    debug_info = {
        "models_dict_exists": models is not None,
        "models_dict_type": str(type(models)),
        "models_keys": list(models.keys()) if models else [],
        "model_exists": "model" in models if models else False,
        "model_type": str(type(models.get("model")))
        if models and "model" in models
        else "None",
        "model_has_predict": hasattr(models.get("model"), "predict")
        if models and "model" in models
        else False,
        "model_has_predict_proba": hasattr(models.get("model"), "predict_proba")
        if models and "model" in models
        else False,
    }

    # Check file existence
    model_file = "models/heart_disease_model_20251207_155704.pkl"
    debug_info["model_file_exists"] = os.path.exists(model_file)

    if os.path.exists(model_file):
        debug_info["model_file_size"] = os.path.getsize(model_file)

    return jsonify(debug_info)


# Add this function to manually reload models
@app.route("/reload-models")
def reload_models():
    """Manually reload models"""
    global models
    print("\n🔄 Manually reloading models...")
    models = load_models()

    if models and models.get("model"):
        return jsonify(
            {
                "status": "success",
                "message": "Models reloaded successfully",
                "model_type": str(type(models.get("model"))),
            }
        )
    else:
        return jsonify({"status": "failed", "message": "Failed to reload models"})


def load_diet_plan():
    """Load diet plan JSON from data directory"""
    try:
        with open("data/Final_diet_plan.json", "r") as f:
            diet_data = json.load(f)
            print("✓ Diet plan loaded successfully")
            return diet_data
    except FileNotFoundError:
        print("✗ Diet plan file not found: data/Final_diet_plan.json")
        return None
    except Exception as e:
        print(f"✗ Error loading diet plan: {e}")
        return None


# Global variables
models = load_models()
diet_plan = load_diet_plan()


def predict_patient_risk(model, patient_data):
    """
    Predict heart disease risk using the loaded model

    Parameters:
    model: trained machine learning model (already loaded)
    patient_data: list of patient features

    Returns:
    dict: prediction results
    """
    try:
        # Convert to numpy array and reshape for prediction
        patient_array = np.array(patient_data).reshape(1, -1)

        # Get prediction probability
        probabilities = model.predict_proba(patient_array)[0]

        # Handle binary classification (assuming class 1 is positive for heart disease)
        if len(probabilities) == 2:
            risk_prob = probabilities[1] * 100  # probability of class 1 (disease)
        else:
            risk_prob = max(probabilities) * 100

        # Get prediction
        prediction = model.predict(patient_array)[0]

        # Determine confidence level based on probability
        max_prob = max(probabilities) * 100
        if max_prob > 80:
            confidence = "High"
        elif max_prob > 65:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Determine risk level and color based on disease probability
        if risk_prob > 70:
            risk_level = "High Risk"
            risk_color = "danger"
            risk_category = "high_risk"
        elif risk_prob > 30:
            risk_level = "Moderate Risk"
            risk_color = "warning"
            risk_category = "moderate_risk"
        else:
            risk_level = "Low Risk"
            risk_color = "success"
            risk_category = "low_risk"

        return {
            "risk_probability": round(risk_prob, 2),
            "risk_level": risk_level,
            "risk_category": risk_category,
            "risk_color": risk_color,
            "prediction": int(prediction),
            "confidence_level": confidence,
            "all_probabilities": [round(p * 100, 2) for p in probabilities],
        }
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None


@app.route("/")
def index():
    """Home page"""
    return render_template("index.html")


@app.route("/register-diet-session", methods=["POST"])
def register_diet_session():
    """Register a new diet chart viewing session"""
    try:
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = {
            "created": datetime.now(),
            "last_ping": datetime.now(),
            "images_folder": "static/images/diet_images",
        }

        # Start cleanup monitoring for this session
        threading.Thread(
            target=monitor_session, args=(session_id,), daemon=True
        ).start()

        return jsonify(
            {
                "status": "success",
                "session_id": session_id,
                "message": "Diet session registered",
            }
        )
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/ping-session", methods=["POST"])
def ping_session():
    """Keep session alive with periodic pings"""
    try:
        data = request.get_json()
        session_id = data.get("session_id")

        if session_id in active_sessions:
            active_sessions[session_id]["last_ping"] = datetime.now()
            return jsonify({"status": "success", "message": "Session pinged"})
        else:
            return jsonify({"status": "error", "message": "Session not found"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/cleanup-images", methods=["POST"])
def cleanup_images():
    """Immediate cleanup when user explicitly leaves"""
    global crawling_status
    try:
        data = request.get_json() or {}
        session_id = data.get("session_id")

        # If session_id provided, mark it for cleanup
        if session_id and session_id in active_sessions:
            active_sessions[session_id]["cleanup_requested"] = True

        # Always attempt cleanup
        images_dir = "static/images/diet_images"
        if os.path.exists(images_dir):

            def delete_folder():
                try:
                    time.sleep(1)  # Small delay
                    if os.path.exists(images_dir):
                        shutil.rmtree(images_dir)
                        print(f"✓ Deleted diet images folder: {images_dir}")

                        # Clear crawling status cache
                        crawling_status.clear()
                        print("✓ Cleared crawling status cache")

                        # Clear all active sessions
                        active_sessions.clear()

                except Exception as e:
                    print(f"Error deleting folder: {e}")

            threading.Thread(target=delete_folder, daemon=True).start()
            return jsonify(
                {"status": "cleanup_started", "message": "Image cleanup initiated"}
            )
        else:
            return jsonify(
                {"status": "no_folder", "message": "No images folder to cleanup"}
            )

    except Exception as e:
        print(f"Error in cleanup: {e}")
        return jsonify({"status": "error", "message": str(e)})


def monitor_session(session_id):
    """Monitor session and cleanup if inactive for too long"""
    global crawling_status
    try:
        while session_id in active_sessions:
            session_data = active_sessions[session_id]
            last_ping = session_data["last_ping"]

            # Check if session is inactive for more than 30 seconds
            if datetime.now() - last_ping > timedelta(seconds=600):
                print(f"Session {session_id} inactive for >600s, cleaning up...")

                # Cleanup images
                images_dir = session_data["images_folder"]
                if os.path.exists(images_dir):
                    try:
                        shutil.rmtree(images_dir)
                        print(f"✓ Auto-cleanup: Deleted {images_dir}")

                        crawling_status.clear()

                    except Exception as e:
                        print(f"Auto-cleanup error: {e}")

                # Remove session
                active_sessions.pop(session_id, None)
                break

            # Check if explicit cleanup was requested
            if session_data.get("cleanup_requested"):
                print(f"Session {session_id} cleanup requested, cleaning up...")

                images_dir = session_data["images_folder"]
                if os.path.exists(images_dir):
                    try:
                        shutil.rmtree(images_dir)
                        print(f"✓ Requested cleanup: Deleted {images_dir}")

                        crawling_status.clear()

                    except Exception as e:
                        print(f"Requested cleanup error: {e}")

                active_sessions.pop(session_id, None)
                break

            # Sleep for 5 seconds before next check
            time.sleep(5)

    except Exception as e:
        print(f"Session monitoring error: {e}")
        active_sessions.pop(session_id, None)


# Cleanup old sessions periodically
def cleanup_old_sessions():
    """Clean up sessions older than 5 minutes"""
    try:
        current_time = datetime.now()
        sessions_to_remove = []

        for session_id, data in active_sessions.items():
            if current_time - data["created"] > timedelta(minutes=5):
                sessions_to_remove.append(session_id)

        for session_id in sessions_to_remove:
            print(f"Removing old session: {session_id}")
            active_sessions.pop(session_id, None)

    except Exception as e:
        print(f"Old session cleanup error: {e}")


# Run cleanup every 2 minutes
def schedule_cleanup():
    while True:
        time.sleep(120)  # 2 minutes
        cleanup_old_sessions()


# Start the background cleanup scheduler
threading.Thread(target=schedule_cleanup, daemon=True).start()


def cleanup_on_exit():
    """Clean up images folder when server shuts down"""
    global crawling_status
    try:
        images_dir = "static/images/diet_images"
        if os.path.exists(images_dir):
            shutil.rmtree(images_dir)
            print("✓ Server shutdown: Cleaned up diet images folder")
        active_sessions.clear()
        crawling_status.clear()
    except Exception as e:
        print(f"Shutdown cleanup error: {e}")


atexit.register(cleanup_on_exit)

DEFAULT_VALUES = {
    "gender": 1,  # Female
    "smoking": 0,  # No
    "alcohol": 0,  # No
    "active": 1,  # Active lifestylea
    "glucose": 1,
}


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Heart disease prediction page with primary features and optional advanced features"""
    if request.method == "GET":
        # Check if there's an existing prediction in session
        if "prediction_result" in session:
            return render_template(
                "predict.html",
                existing_prediction=session["prediction_result"],
                default_values=DEFAULT_VALUES,
            )
        else:
            return render_template("predict.html", default_values=DEFAULT_VALUES)

    try:
        # Get primary features (required) - Top 6 by importance
        age_years = float(request.form["age"])  # Age in years (will convert to days)
        weight = float(request.form["weight"])  # Weight in kg
        height = float(request.form["height"])  # Height in cm
        ap_hi = float(request.form["ap_hi"])  # Systolic BP
        ap_lo = float(request.form["ap_lo"])  # Diastolic BP
        cholesterol = int(
            request.form["cholesterol"]
        )  # 1=normal, 2=above, 3=well above

        # Get optional features with defaults (less important)
        gluc = int(request.form.get("gluc", DEFAULT_VALUES["glucose"]))
        smoke = int(request.form.get("smoke", DEFAULT_VALUES["smoking"]))
        alco = int(request.form.get("alco", DEFAULT_VALUES["alcohol"]))
        active = int(request.form.get("active", DEFAULT_VALUES["active"]))
        gender = int(
            request.form.get("gender", DEFAULT_VALUES["gender"])
        )  # ADD THIS LINE - now required

        # Calculate BMI for display
        bmi = weight / ((height / 100) ** 2)
        # pulse pressure
        pulse_pressure = ap_hi - ap_lo
        # health_index
        health_index = (active * 1) - (smoke * 0.5) - (alco * 0.5)
        # cholestrol-glucose
        cholesterol_gluc_interaction = cholesterol * gluc

        # Create patient data array (adjust order based on your model's expected features)
        patient_data = [
            gender,
            weight,
            ap_hi,
            ap_lo,
            cholesterol,
            gluc,
            smoke,
            alco,
            active,
            age_years,
            bmi,
            pulse_pressure,
            health_index,
            cholesterol_gluc_interaction,
        ]

        # Create user_data_dict for display purposes
        user_data_dict = {
            "age": age_years,
            "gender": "Male" if gender == 2 else "Female",
            "height": height,
            "weight": weight,
            "bmi": round(bmi, 1),
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "cholesterol": ["Normal", "Above Normal", "Well Above Normal"][
                cholesterol - 1
            ],
            "glucose": ["Normal", "Above Normal", "Well Above Normal"][gluc - 1],
            "smoking": "Yes" if smoke == 1 else "No",
            "alcohol": "Yes" if alco == 1 else "No",
            "active": "Yes" if active == 1 else "No",
        }

        print(f"Patient data received: {len(patient_data)} features")

        # Make prediction using the loaded model
        if models and models.get("model"):
            prediction_result = predict_patient_risk(models["model"], patient_data)

            if prediction_result:
                # Save outcome for diet generation
                outcome = {
                    "risk_probability": prediction_result["risk_probability"],
                    "risk_level": prediction_result["risk_level"],
                    "risk_category": prediction_result["risk_category"],
                    "risk_color": prediction_result["risk_color"],
                    "prediction": prediction_result["prediction"],
                    "confidence_level": prediction_result["confidence_level"],
                    "user_data": user_data_dict,
                    "patient_features": patient_data,
                    "timestamp": datetime.now().isoformat(),
                    "feature_count": len(patient_data),
                }

                # Save to session (for persistence across page navigation)
                session["prediction_result"] = outcome
                session.permanent = True

                # Save using pickle (for diet generation)
                try:
                    with open("static/diet_images/outcome.pkl", "wb") as f:
                        pickle.dump(outcome, f)
                    print("✓ Outcome saved successfully")
                except Exception as e:
                    print(f"✗ Error saving outcome: {e}")

                return render_template(
                    "result.html", prediction=outcome, user_data=user_data_dict
                )
            else:
                flash("Error in prediction calculation. Please try again.", "error")
                return redirect(url_for("predict"))
        else:
            flash("Model not loaded properly. Please check model files.", "error")
            return redirect(url_for("predict"))

    except KeyError as e:
        missing_field = str(e).replace("'", "")
        flash(
            f"Missing required field: {missing_field}. Please fill all required fields.",
            "error",
        )
        print(f"Missing form field: {e}")
        return redirect(url_for("predict"))
    except ValueError as e:
        flash(f"Invalid input: {str(e)}", "error")
        print(f"Value error: {e}")
        return redirect(url_for("predict"))
    except Exception as e:
        print(f"Prediction error: {e}")
        print(f"Available form fields: {list(request.form.keys())}")
        flash(f"Error in prediction: {str(e)}", "error")
        return redirect(url_for("predict"))


@app.route("/test-model")
def test_model():
    """Test model with sample data"""
    if not models or not models.get("model"):
        return "Model not loaded!"

    model = models["model"]

    # Sample patient data (all 14 features)
    # age_days, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
    test_data = [1, 85, 140, 90, 3, 1, 0, 0, 1, 55, 349276, 50, 1, 3]

    print("\n" + "=" * 60)
    print("TESTING MODEL WITH SAMPLE DATA")
    print("=" * 60)

    result = predict_patient_risk(model, test_data)

    if result:
        return jsonify(
            {"status": "success", "test_data": test_data, "prediction_result": result}
        )
    else:
        return jsonify(
            {"status": "failed", "message": "Check console for error details"}
        )


@app.route("/diet-plan")
def diet_plan_page():
    """Diet plan generation page"""
    try:
        # First check session for prediction result
        if "prediction_result" in session:
            outcome = session["prediction_result"]
            print(f"Loaded outcome from session: {outcome['risk_category']}")
        else:
            # Fallback to pickle file
            try:
                with open("static/diet_images/outcome.pkl", "rb") as f:
                    outcome = pickle.load(f)
                print(f"Loaded outcome from pickle: {outcome['risk_category']}")
            except FileNotFoundError:
                flash("Please complete heart disease prediction first.", "warning")
                return redirect(url_for("predict"))

        return render_template("diet_plan.html", outcome=outcome)
    except Exception as e:
        print(f"Error loading outcome: {e}")
        flash(f"Error loading prediction results: {str(e)}", "error")
        return redirect(url_for("predict"))


def get_recommendations_by_risk(risk_category):
    """Get detailed recommendations based on risk level"""
    if risk_category == "high_risk":
        return [
            "Follow a strict low-sodium diet with less than 2,300mg sodium per day",
            "Limit saturated fats to less than 10% of total daily calories",
            "Include omega-3 rich foods such as fatty fish, walnuts, and flaxseeds daily",
            "Consume at least 5 servings of fruits and vegetables per day",
            "Choose whole grains over refined grains in all meals",
            "Eliminate or severely limit processed foods, fast foods, and added sugars",
            "Practice strict portion control and consider using smaller plates",
            "Schedule regular follow-ups with your healthcare provider",
        ]
    elif risk_category == "moderate_risk":
        return [
            "Maintain a well-balanced diet with variety from all food groups",
            "Include heart-healthy fats like olive oil, avocados, and nuts daily",
            "Choose lean proteins such as poultry, fish, legumes, and tofu",
            "Consume high-fiber foods including vegetables, fruits, and whole grains",
            "Limit processed and fried foods to occasional treats",
            "Monitor portion sizes and practice mindful eating",
            "Stay adequately hydrated with water as your primary beverage",
            "Consider meal timing and avoid late-night eating",
        ]
    else:  # low_risk
        return [
            "Continue maintaining healthy eating patterns as prevention",
            "Include a colorful variety of fruits and vegetables in your daily meals",
            "Choose lean proteins and vary your protein sources throughout the week",
            "Stay physically active and maintain a healthy weight",
            "Practice good hydration habits with adequate water intake",
            "Use mindful eating practices and enjoy your meals",
            "Plan and prepare meals ahead of time when possible",
            "Schedule regular health check-ups to monitor your continued wellness",
        ]




@app.route("/about")
def about():
    """About page"""
    return render_template("about.html")


@app.route("/contact")
def contact():
    """Contact page"""
    return render_template("contact.html")


@app.route("/health")
def health_check():
    """Health check endpoint"""
    global crawling_status
    status = {
        "status": "healthy",
        "model_loaded": bool(models and models.get("model")),
        "diet_plan_loaded": bool(diet_plan),
        "image_crawler": "Bing Image Crawler",
        "crawling_status": {
            "total_cached_images": len(
                [s for s in crawling_status.values() if s != "crawling"]
            ),
            "currently_crawling": len(
                [s for s in crawling_status.values() if s == "crawling"]
            ),
            "cache_size": len(crawling_status),
        },
        "timestamp": datetime.now().isoformat(),
    }
    return jsonify(status)


# Add favicon route to prevent 404 errors
@app.route("/favicon.ico")
def favicon():
    try:
        return send_file("static/favicon.ico", mimetype="image/vnd.microsoft.icon")
    except:
        return "", 204  # No content response if favicon not found


@app.errorhandler(404)
def not_found_error(error):
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Page Not Found - CarePulse</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                margin-top: 100px; 
                background-color: #f8f9fa;
            }
            .container {
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 { color: #dc3545; }
            .btn {
                display: inline-block;
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>404 - Page Not Found</h1>
            <p>The page you are looking for doesn't exist.</p>
            <a href="/" class="btn">Return to Home</a>
            <a href="/predict" class="btn">Heart Disease Prediction</a>
        </div>
    </body>
    </html>
    """), 404


@app.errorhandler(500)
def internal_error(error):
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Server Error - CarePulse</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                margin-top: 100px; 
                background-color: #f8f9fa;
            }
            .container {
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 { color: #dc3545; }
            .btn {
                display: inline-block;
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>500 - Internal Server Error</h1>
            <p>Something went wrong on our end. Please try again later.</p>
            <a href="/" class="btn">Return to Home</a>
        </div>
    </body>
    </html>
    """), 500


if __name__ == "__main__":
    print("=" * 60)
    print("🏥 Starting CarePulse Flask Application with Bing Image Crawler...")
    print("=" * 60)

    # Create placeholder image
    create_placeholder_image()

    # Check if required files exist
    required_files = [
        "models/heart_disease_model_20251207_155704.pkl",
        "data/Final_diet_plan.json",
    ]

    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ Found: {file_path}")
        else:
            missing_files.append(file_path)
            print(f"✗ Missing: {file_path}")

    # Status summary
    print("\n📊 Application Status:")
    print(f"   Models loaded: {'✓ Yes' if models and models.get('model') else '✗ No'}")
    print(f"   Diet plan loaded: {'✓ Yes' if diet_plan else '✗ No'}")
    print(f"   Image crawler: ✓ Bing Image Crawler")

    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("   Please ensure all required files are in place.")
    else:
        print("\n✅ All required files found!")

    print(f"\n🚀 Server will start at: http://localhost:5000")
    print("   Available endpoints:")
    print("   - / (Home)")
    print("   - /predict (Heart Disease Prediction)")
    print("   - /diet-plan (Diet Plan Generation)")
    print("   - /health (Health Check)")
    print("   - /get-food-image/<food_name> (Food Image API)")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
