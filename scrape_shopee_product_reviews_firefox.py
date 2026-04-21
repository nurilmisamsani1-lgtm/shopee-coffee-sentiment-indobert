import time
import random
import pandas as pd
import os

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


OUTPUT_FILE = "shopee_product_reviews.csv"

options = Options()
options.debugger_address = "127.0.0.1:9222"

driver = webdriver.Firefox(options=options)
wait = WebDriverWait(driver, 15)

seen_ids = set()
page = 1


while True:

    print(f"Scraping page {page}")

    wait.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.q2b7Oq"))
    )

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(2)

    reviews = driver.find_elements(By.CSS_SELECTOR, "div.q2b7Oq")

    page_reviews = []

    for r in reviews:

        review_id = r.get_attribute("data-cmtid")

        if review_id in seen_ids:
            continue

        seen_ids.add(review_id)

        username = ""
        u = r.find_elements(By.CSS_SELECTOR, ".InK5kS")
        if u:
            username = u[0].text

        comment = ""
        c = r.find_elements(By.CSS_SELECTOR, ".YNedDV")
        if c:
            comment = c[0].text

        date_variant = ""
        d = r.find_elements(By.CSS_SELECTOR, ".XYk98l")
        if d:
            date_variant = d[0].text

        rating = len(r.find_elements(By.CSS_SELECTOR, "svg.icon-rating-solid"))

        attributes = {}
        attr_rows = r.find_elements(By.CSS_SELECTOR, ".meQyXP > div > div")

        for row in attr_rows:
            spans = row.find_elements(By.TAG_NAME, "span")
            if len(spans) >= 2:
                key = spans[0].text.replace(":", "").strip()
                value = spans[1].text.strip()
                attributes[key] = value

        rasa = attributes.get("Rasa", "")
        kualitas = attributes.get("Kualitas", "")
        kadaluarsa = attributes.get("Kadaluarsa", "")

        likes = ""
        like_el = r.find_elements(By.CSS_SELECTOR, ".shopee-product-rating__like-count")
        if like_el:
            likes = like_el[0].text

        images = r.find_elements(By.CSS_SELECTOR, ".rating-media-list__image-wrapper")
        videos = r.find_elements(By.CSS_SELECTOR, "video")

        image_count = len(images)
        video_count = len(videos)

        page_reviews.append({
            "review_id": review_id,
            "username": username,
            "rating": rating,
            "date_variant": date_variant,
            "rasa": rasa,
            "kualitas": kualitas,
            "kadaluarsa": kadaluarsa,
            "likes": likes,
            "image_count": image_count,
            "video_count": video_count,
            "comment": comment
        })


    # STOP if no new reviews found
    if len(page_reviews) == 0:
        print("No new reviews found → stopping.")
        break


    df = pd.DataFrame(page_reviews)

    df.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not os.path.exists(OUTPUT_FILE),
        index=False
    )

    print(f"Saved {len(df)} reviews")


    try:

        next_button = driver.find_element(
            By.CSS_SELECTOR,
            "button.shopee-icon-button--right"
        )

        driver.execute_script("arguments[0].click();", next_button)

        page += 1

        time.sleep(random.uniform(2,4))

    except:
        break


print("Scraping finished.")