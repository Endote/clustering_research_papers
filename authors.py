import os
import time
import csv
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

def setup_driver(headless: bool = True):
    """Set up undetected-chromedriver with basic options."""
    options = uc.ChromeOptions()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return uc.Chrome(options=options)

def get_department_urls(driver, start_url):
    """
    Open the starting department page and extract all department URLs from the navigation.
    Looks for a <nav> element with role "navigation" that does not have the classes
    'stellarnav', 'dark', 'right', or 'desktop', then finds all <a> tags within it.
    """
    driver.get(start_url)
    time.sleep(2)  # Wait for page to load
    nav_selector = "nav[role='navigation']:not(.stellarnav):not(.dark):not(.right):not(.desktop) ul li a"
    dept_link_elements = driver.find_elements(By.CSS_SELECTOR, nav_selector)
    dept_urls = [elem.get_attribute("href") for elem in dept_link_elements if elem.get_attribute("href")]
    return list(set(dept_urls))  # Remove duplicates if any

def scrape_department_members(driver, dept_url):
    """
    Open the given department page in a new tab and scrape all faculty members' information.
    It extracts the three rows of data (name, title, email) from each faculty card.
    """
    profiles = []

    driver.get(dept_url)
    time.sleep(2)
    
    # Scrape faculty names.
    faculty_names_elements = driver.find_elements(By.CSS_SELECTOR, "span.largeText")
    faculty_names = [x.text.strip() for x in faculty_names_elements]
    print(faculty_names)
        
    # Find all faculty profile cards on the department page.
    faculty_cards = driver.find_elements(By.CSS_SELECTOR, "div.ccm-profile-member")
    print(f"Found {len(faculty_cards)} faculty cards on {dept_url}")
    
    for i, card in enumerate(faculty_cards):
        name = faculty_names[i]
        try:
            fields = card.find_elements(By.CSS_SELECTOR, ".ccm-profile-member-fields-data")
            area = fields[0].text.strip() 
            title = fields[1].text.strip()
            email = fields[2].text.strip()
            
        except Exception:
            title, email = "ERROR", "ERROR"
        
        profile = {
            "Name": name,
            "Title": title,
            "Email": email,
            "Department": dept_url,
            "Area": area

        }
        profiles.append(profile)
        print(f"Scraped: {name} | {title} | {email}")
    
    # Close the new tab and switch back to the main window.
    # driver.close()
    return profiles

def save_profiles_csv(profiles, filename="faculty_profiles.csv"):
    """Save the list of profile dictionaries to a CSV file."""
    csv_columns = ["Name", "Title", "Email", "Department", "Area"]
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    with open(filepath, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(profiles)
    print(f"Saved profiles to {filepath}")

def main():
    # Starting department page URL.
    start_url = "https://www.wne.uw.edu.pl/wydzial/struktura-wydzialu/katedry-i-zaklady/katedra-data-science"
    driver = setup_driver(headless=False)
    
    try:
        # Extract all department URLs from the navigation.
        department_urls = get_department_urls(driver, start_url)
        print("Found department URLs:")
        print(department_urls)
        
        all_profiles = []
        # For each department, open in a new tab, scrape the profiles, and add to the list.
        for dept_url in department_urls:
            print(f"\nScraping department: {dept_url.split('/')[-1]}")
            driver.get(dept_url)
            dept_profiles = scrape_department_members(driver, dept_url)
            print(dept_profiles)
            all_profiles.extend(dept_profiles)
            print("Current profiles:", all_profiles)
        
        # Uncomment the following line to save the profiles to a CSV file.
        save_profiles_csv(all_profiles)
    
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()
