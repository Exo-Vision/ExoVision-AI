import requests
from bs4 import BeautifulSoup

base_url = "https://science.nasa.gov/exoplanets/exoplanet-catalog/"
page = 1

while True:
    resp = requests.get(base_url, params={"page": page})
    if resp.status_code != 200 or "No results" in resp.text:
        break
    
    soup = BeautifulSoup(resp.text, "html.parser")
    elements = soup.find_all(class_="hds-a11y-heading-22")
    if not elements:
        break
    
    for el in elements:
        print(el.get_text(strip=True))
    
    page += 1
