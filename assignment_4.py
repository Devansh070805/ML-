
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

def scrape_books_to_csv():
    base_url = "https://books.toscrape.com/"
    url = base_url
    titles = []
    prices = []
    availability_list = []
    ratings = []
    rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    while True:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        books = soup.find_all("article", class_="product_pod")
        for book in books:
            title = book.h3.a["title"].strip()
            price = book.find("p", class_="price_color").get_text(strip=True)
            availability = book.find("p", class_="instock availability").get_text(strip=True)
            star_tag = book.find("p", class_="star-rating")
            star_classes = star_tag.get("class", [])
            rating_word = None
            for c in star_classes:
                if c in rating_map:
                    rating_word = c
                    break
            rating_value = rating_map.get(rating_word, None)
            titles.append(title)
            prices.append(price)
            availability_list.append(availability)
            ratings.append(rating_value)
        next_li = soup.find("li", class_="next")
        if next_li and next_li.a:
            next_href = next_li.a["href"]
            url = urljoin(url, next_href)
        else:
            break
    df = pd.DataFrame(
        {
            "Title": titles,
            "Price": prices,
            "Availability": availability_list,
            "StarRating": ratings,
        }
    )
    df.to_csv("books.csv", index=False)

def scrape_imdb_top250_to_csv():
    url = "https://www.imdb.com/chart/top/"
    driver = webdriver.Chrome()
    driver.get(url)
    time.sleep(5)
    rows = driver.find_elements(By.CSS_SELECTOR, "table.chart.full-width tr")
    ranks = []
    titles = []
    years = []
    ratings = []
    for row in rows:
        try:
            title_column = row.find_element(By.CSS_SELECTOR, "td.titleColumn")
            rating_column = row.find_element(By.CSS_SELECTOR, "td.ratingColumn.imdbRating")
        except Exception:
            continue
        text_full = title_column.text
        parts = text_full.split("\n")
        if len(parts) < 2:
            continue
        first_line = parts[0]
        title_line = parts[1]
        rank_str = first_line.split(".")[0].strip()
        try:
            rank = int(rank_str)
        except ValueError:
            continue
        title = title_line.strip()
        try:
            year_span = title_column.find_element(By.TAG_NAME, "span").text
            year = year_span.strip("() ")
        except Exception:
            year = ""
        try:
            rating = rating_column.find_element(By.TAG_NAME, "strong").text.strip()
        except Exception:
            rating = ""
        ranks.append(rank)
        titles.append(title)
        years.append(year)
        ratings.append(rating)
    driver.quit()
    df = pd.DataFrame(
        {
            "Rank": ranks,
            "Title": titles,
            "Year": years,
            "IMDBRating": ratings,
        }
    )
    df.to_csv("imdb_top250.csv", index=False)

def scrape_weather_to_csv():
    url = "https://www.timeanddate.com/weather/"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="zebra tb-wt fw va-m")
    if table is None:
        table = soup.find("table", id="wt-48")
    cities = []
    temps = []
    conditions = []
    if table is not None:
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue
            city_tag = cells[0].find("a")
            city = city_tag.get_text(strip=True) if city_tag else cells[0].get_text(strip=True)
            temp = cells[1].get_text(strip=True)
            condition = cells[2].get_text(strip=True)
            cities.append(city)
            temps.append(temp)
            conditions.append(condition)
    df = pd.DataFrame(
        {
            "City": cities,
            "Temperature": temps,
            "Condition": conditions,
        }
    )
    df.to_csv("weather.csv", index=False)

if __name__ == "__main__":
    scrape_books_to_csv()
    scrape_imdb_top250_to_csv()
    scrape_weather_to_csv()
