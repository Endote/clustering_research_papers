import os
import time
import random
import json
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException, InvalidSessionIdException

import bertopic_cluster


def setup_driver(headless: bool = True):
    """Set up Chrome WebDriver with undetected-chromedriver."""
    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return uc.Chrome(options=options)


def read_authors(file_path):
    """Read author names from a file."""
    with open(file_path, 'r', encoding="utf-8") as file:
        return [line.strip() for line in file.readlines()]


def save_paper(author, paper, save_dir="crawled_data"):
    """Save a single paper's data to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    author_file = os.path.join(save_dir, f"{author.replace(' ', '_')}.json")

    # Load existing data or create new list
    if os.path.exists(author_file):
        with open(author_file, "r", encoding="utf-8") as file:
            data = json.load(file)
    else:
        data = []

    # Append the new paper and save
    data.append(paper)
    with open(author_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def search_author(driver, author_name):
    """Search for an author on Google Scholar."""
    driver.get("https://scholar.google.com")
    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys(author_name)
    search_box.send_keys(Keys.RETURN)
    time.sleep(random.uniform(2, 4))

    try:
        profile_link = driver.find_element(By.CSS_SELECTOR, "h4.gs_rt2 a")
        return profile_link.get_attribute("href")
    except NoSuchElementException:
        print(f"Profile not found for author: {author_name}")
        return None


def crawl_author_papers(driver, profile_url, author_name):
    """Crawl all papers for an author."""
    driver.get(profile_url)
    time.sleep(random.uniform(2, 4))

    try:
        paper_elements = driver.find_elements(By.CSS_SELECTOR, ".gsc_a_tr")
        for paper in paper_elements:
            try:
                # Open the paper page
                title_element = paper.find_element(By.CSS_SELECTOR, ".gsc_a_at")
                title = title_element.text
                paper_url = title_element.get_attribute("href")
                ActionChains(driver).key_down(Keys.CONTROL).click(title_element).key_up(Keys.CONTROL).perform()
                driver.switch_to.window(driver.window_handles[-1])
                time.sleep(random.uniform(2, 4))

                # Extract paper details
                authors = None
                year = None
                abstract = None
                keywords = None
                correct_url = None

                try:
                    authors = driver.find_element(By.XPATH, "//div[.='Autorzy']/following-sibling::div").text
                except NoSuchElementException:
                    pass

                try:
                    year = driver.find_element(By.XPATH, "//div[.='Data publikacji']/following-sibling::div").text
                except NoSuchElementException:
                    pass

                try:
                    abstract = driver.find_element(By.ID, "gsc_oci_descr").text
                except NoSuchElementException:
                    pass

                try:
                    correct_url = driver.find_element(By.CSS_SELECTOR, "#gsc_oci_title a").get_attribute("href")
                except NoSuchElementException:
                    pass

                paper_data = {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "abstract": abstract,
                    "keywords": keywords,  # Placeholder, not found in HTML snippet
                    "url": correct_url or paper_url,
                }
                save_paper(author_name, paper_data)

                # Close the paper tab and return to main
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
            except Exception as e:
                print(f"Error processing a paper for {author_name}: {e}")
    except Exception as e:
        print(f"Error crawling papers for {author_name}: {e}")


def main():
    authors = read_authors("list.txt")
    save_dir = "crawled_data"

    for author in authors:
        print(f"Processing author: {author}")
        driver = None

        try:
            driver = setup_driver()
            profile_url = search_author(driver, author)

            if profile_url:
                crawl_author_papers(driver, profile_url, author)
        except InvalidSessionIdException:
            print(f"Session invalid for author {author}, restarting driver...")
            if driver:
                driver.quit()
        except Exception as e:
            print(f"Unexpected error for {author}: {e}")
        finally:
            if driver:
                driver.quit()

    print(f"Crawling complete. Data saved in '{save_dir}'")


if __name__ == "__main__":
    main()
