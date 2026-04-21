import time
import random
import pandas as pd
import os

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


OUTPUT_FILE = "shopee_reviews.csv"

options = Options()
options.debugger_address = "127.0.0.1:9222"

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

seen_ids = set()

page = 1

while True:

    print(f"Scraping page {page}")

    wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.q2b7Oq"))
    )

    # scroll to ensure all reviews load
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(1)

    reviews = driver.find_elements(By.CSS_SELECTOR, "div.q2b7Oq")

    page_reviews = []
    stop_scraping = False

    for r in reviews:

        review_id = r.get_attribute("data-cmtid")

        if review_id in seen_ids:
            stop_scraping = True
            break

        seen_ids.add(review_id)

        # USERNAME (works for anonymous and normal users)
        username_el = r.find_elements(By.CSS_SELECTOR, ".InK5kS")
        username = username_el[0].text if username_el else ""

        # COMMENT
        comment_el = r.find_elements(By.CSS_SELECTOR, "div.YNedDV")
        comment = comment_el[0].text if comment_el else ""

        # DATE + VARIANT
        date_variant_el = r.find_elements(By.CSS_SELECTOR, "div.XYk98l")
        date_variant = date_variant_el[0].text if date_variant_el else ""

        # PRODUCT NAME
        product_el = r.find_elements(By.CSS_SELECTOR, "span.EQ3yLe")
        product_name = product_el[0].text if product_el else ""

        # RATING
        rating = len(r.find_elements(By.CSS_SELECTOR, "svg.icon-rating-solid"))

        page_reviews.append({
            "review_id": review_id,
            "product_name": product_name,
            "username": username,
            "rating": rating,
            "date_variant": date_variant,
            "comment": comment
        })

    # Save results immediately
    df = pd.DataFrame(page_reviews)

    if not df.empty:
        df.to_csv(
            OUTPUT_FILE,
            mode="a",
            header=not os.path.exists(OUTPUT_FILE),
            index=False
        )

        print(f"Saved {len(df)} reviews")

    if stop_scraping:
        print("Duplicate detected → stopping.")
        break

    try:
        next_button = driver.find_element(
            By.CSS_SELECTOR,
            "button.shopee-icon-button--right"
        )

        driver.execute_script("arguments[0].click();", next_button)

        page += 1

        # random delay to avoid detection
        time.sleep(random.uniform(2,4))

    except:
        break


print("Scraping finished.")