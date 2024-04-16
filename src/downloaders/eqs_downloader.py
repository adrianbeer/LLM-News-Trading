import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime as dt
import xlwings as xw
import glob
from pyhtml2pdf import converter # here might occur errors becuause library isnt stable

# ####### Categories
# Corporate
# Vorabbekanntmachung
# Directors Dealings
# Hauptversammlung
# Ad-hoc
# Gesamtstimmrechte
# Media
# Stimmrechtsanteile
# Kapitalmarktinformationen
# Press Release == UK Regulatory
# UK Regulatory
# Übernahmeangebot
# Related Party Transactions
# FR Regulatory


month_dict = {"Januar": 1,
            "Februar": 2,
            "März": 3,
            "April": 4,
            "Mai": 5,
            "Juni": 6,
            "Juli": 7,
            "August": 8,
            "September": 9,
            "Oktober": 10,
            "November": 11,
            "Dezember": 12}

class EQS_Scraper():
    def __init__(self, isin: str, save_pdf: bool, save_csv: bool,
                 path_save: str, num_to_scrape: int, month_dict: dict):
        self.isin = isin
        self.save_pdf = save_pdf
        self.save_csv = save_csv
        self.path_save = path_save
        self.num_to_scrape = num_to_scrape
        self.month_dict = month_dict

    def scrape_inital_link(self) -> list:
        """Scrapes the base link provided and returns a list filled with
           the release links, the release date and the release type for each
           found single news release

        Returns:
            list: 3 lists
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:100.0) Gecko/20100101 Firefox/100.0"
        }
        initial_url = f"https://www.eqs-news.com/de/search-results/?searchtype=news&searchword={self.isin}"
        print(f"Scraping initial url for isin: {self.isin}")
        initial_soup = BeautifulSoup(requests.get(initial_url, headers=headers).content, "html.parser")

        num_all_releases = int(initial_soup.find("p", {"class": "search-news__results"}).text.strip().split()[-2])
        print(f"Found {num_all_releases} news releases for isin: {self.isin}")
        # Edit the url to get the max number of results
        url = f"https://www.eqs-news.com/de/search-results/?searchtype=news&searchword={self.isin}&searchphrase=all&ordering=newest&pageLimit={num_all_releases}"
        soup = BeautifulSoup(requests.get(url, headers=headers).content, "html.parser")

        # Scrape all the single links to the very own news pages that should be scraped
        # Here we have to add also a clicker that clicks itslef through the single results pages
        news_links = []
        for link in soup.find("div", {"class": "search-news"}).find_all("a"):
            news_links.append(link.get('href'))
        # Also save the publication date of the news release
        release_dates = []
        for date in soup.find_all("div", {"class": "col-auto search-news-grid__date"}):
            release_dates.append(date.text)
        # Also get the type of news release
        release_type = []
        for r_type in soup.find_all("div", {"class": "col-auto search-news-grid__type"}):
            release_type.append(r_type.text)

        print(f"Found {len(news_links)} links for news releases")
        if num_to_scrape:
            news_links = news_links[:self.num_to_scrape]
            release_dates = release_dates[:self.num_to_scrape]
            release_type = release_type[:self.num_to_scrape]

        return news_links, release_dates, release_type


    def getSoupFromLink(self, r_link: list) -> list:
        """Creates the soup (html) for a provided list of links for each one
           and returns a list of the respective soup objects

        Args:
            r_link (list): lits of links for which to get the souo objects

        Returns:
            list: list of soup objects
        """
        # For a provided link from the initial scrape get the single soups
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:100.0) Gecko/20100101 Firefox/100.0"
        }
        soup_list = []
        len_soup_list = len(r_link)
        for ind, link in enumerate(r_link):
            print(f"Creating soup object for link: {link}, progress: {round((ind/len_soup_list)*100, 1)} %")
            soup = BeautifulSoup(requests.get(link, headers=headers).content, "html.parser")
            soup_list.append(soup)
        return soup_list


    def getTitle_CompanyName(self, soup_article: BeautifulSoup, r_link: str,
                             r_date: str, r_type: str) -> dict:
        """Extracts the Company Name and the Release Title for a given link out
           of its provided BeautifoulSoup object

        Args:
            soup_article (BeautifulSoup): Soup of the given link
            r_link (str): single link of the news release
            r_date (str): single release date
            r_type (str): single release type

        Returns:
            dict: title and error dict
        """
        # Get the title element
        title_dict = {"Company_Name": None,
                      "Release_Title": None}
        release_title_err = []
        # If the element is available try to get it
        if soup_article.find('h1', {"class": "news-details__title pb-2"}):
            try:
                release_title = soup_article.find('h1', {"class": "news-details__title pb-2"}).text.strip().split("\n")
            except:
                print(f"Problems getting release title at: {r_link}")
                release_title_err.append({"isin": self.isin,
                                          "link": r_link,
                                          "type": r_type})
        try:
            title_dict["Company_Name"] = release_title[0]
            title_dict["Release_Title"] = release_title[1]
        except:
            print(f"Problems saving title elements in dict at: {r_link}")
            release_title_err.append({"isin": self.isin,
                                      "link": r_link,
                                      "type": r_type})
        return title_dict, release_title_err


    def getHeader_Meta_Data(self, soup_article: BeautifulSoup, r_link: str,
                            r_date: str, r_type: str) -> list:
        """Extracts the Meta Data from the Header section of the single respective
           News Release

        Args:
            soup_article (BeautifulSoup): Soup of the given link
            r_link (str): single link of the news release
            r_date (str): single release date
            r_type (str): single release type

        Returns:
            list: _description_
        """
        header_meta_data_dict = {"Company_Name": None,
                                 "Release_Title": None,
                                 "Release_Date": None}
        header_meta_data_err = []
        release_header = []
        try:
            # in some cases we have a dedicated news_top section we can parse but not for every release type
            if soup_article.find('p', {"class": "news_top"}):
            #if r_type in ["Vorabbekanntmachung",
            #              "Hauptversammlung",
            #              "Ad-hoc",
            #              "Media",
            #              "Stimmrechtsanteile",
            #              "Kapitalmarktinformationen",
            #              "Press Release",
            #              "UK Regulatory"]:
                for line in soup_article.find('p', {"class": "news_top"}):
                    release_header.append(line.text.strip().replace("\n", ""))
                header_meta_data = [item for item in release_header if item != ""]
            else:  # would fail at: Übernahmeangebot
                header_meta_data = soup_article.find('div', {"class": "news-details__content"}).find_all("p")[0].text.strip().split("\n")
                # Drop all the empty list items
                header_meta_data = [item for item in header_meta_data if item != ""]
        except:
            print(f"Problems with Header Meta Data at: {r_link}")
            header_meta_data_err.append({"isin": self.isin,
                                            "link": r_link,
                                            "type": r_type})
        return release_header, header_meta_data_err


    def getReleaseBodyTitle(self, soup_article: BeautifulSoup, r_link: str) -> list:
        """Extracts the Body Title of the News Release

        Args:
            soup_article (BeautifulSoup): Soup of the given link
            r_link (str): single link of the news release

        Returns:
            list: Release Body Title
        """
        release_body_title = []
        try:
            release_body_title = soup_article.find('div', {"class": "news-details__content"}).find_all("p")[1].text.strip()
        except:
            print(f"Problems Release Body Title at: {r_link}")
        return release_body_title


    def getReleaseDiscText(self, soup_article: BeautifulSoup, r_link: str) -> list:
        """Extracts the Description Text, unfortunately this section is not
           always available, therefore it has to be handled with care

        Args:
            soup_article (BeautifulSoup): Soup of the given link
            r_link (str): single link of the news release

        Returns:
            list: Optional News Release Description Text
        """
        # not always existend in the body of the website, the short description slot
        disc_text = []
        if soup_article.find_all("ul", {"type": "disc"}):
            for l_i in soup_article.find_all("ul", {"type": "disc"}):
                disc_text.append(l_i.text)
            disc_text = " ".join(disc_text)
        else:
            print(f"No Release Description Text available for: {r_link}")
        return disc_text


    def getReleaseBodyText(self, soup_article: BeautifulSoup, r_link: str,
                           r_date: str, r_type: str) -> list:
        """Extracts the main body text from the Release

        Args:
            soup_article (BeautifulSoup): Soup of the given link
            r_link (str): single link of the news release
            r_date (str): single release date
            r_type (str): single release type

        Returns:
            list: normal lists as well as error lists
        """
        # Rest of the text
        # the release body lasts until one of the following keywords
        counter_contact = 0
        for line in soup_article.find('div', {"class": "news-details__content"}).find_all("p")[1:]:
            if (("kontakt:") or ("contact:") or ("kontakte") or (" kontakt:")) not in line.text.strip().lower():
                counter_contact +=1
            else:
                counter_contact +=1
                break

        release_text = []
        release_text_err = []
        try:
            if r_type == "Vorabbekanntmachung":
                for line in soup_article.find('div', {"class": "break-word news_main"}):
                    release_text.append(line.text.strip().replace("\n", " "))
            elif r_type in ["Hauptversammlung", "Ad-hoc", "Press Release", "UK Regulatory"]:
                # Next get the main release body
                # Add here an option that searches instead for the p elements for the span elements
                for line in soup_article.find('div', {"class": "break-word news_main"}).find_all("p"):
                    release_text.append(line.text.strip().replace("\n", " "))
            elif r_type == "Media":
                for line in soup_article.find('div', {"class": "break-word news_main"}).find_all("span"):
                    release_text.append(line.text.strip().replace("\n", ""))
            elif r_type == "Vorabbekanntmachung":
                for line in soup_article.find('table', {"class": "news_layout style2015"}).find_all("div"):
                    release_text.append(line.text.strip().replace("\n", ""))
            else:
                for line in soup_article.find('div', {"class": "news-details__content"}).find_all("p")[2:counter_contact]:
                    release_text.append(line.text.strip())
                if release_text == []:
                    for line in soup_article.find('div', {"class": "news-details__content"}).find_all("pre"):
                        release_text.append(line.text.strip())
            # drop all the empty elements in the list
            release_text = [item for item in release_text if item != ""]
        except:
            print(f"Problems Release Text at: {r_link}")
            release_text_err.append({"isin": self.isin,
                                     "link": r_link,
                                     "type": r_type})
        # next get all the text content that is below the release contact details
        release_text_after_contact = []
        release_text_after_contact_err = []
        try:
            for line in soup_article.find('div',
                                          {"class": "news-details__content"}).find_all("p")[counter_contact:]:
                release_text_after_contact.append(line.text.strip())
        except:
            print(f"Problems Release Text after contacts at: {r_link}")
            release_text_after_contact_err.append({"isin": self.isin,
                                                   "link": r_link,
                                                   "type": r_type})

        return release_text, release_text_err, release_text_after_contact, release_text_after_contact_err


    def getMetaDataBottom(self, soup_article: BeautifulSoup, r_link: str,
                          r_date: str, r_type: str) -> dict:
        """Extracts Meta Data from the bottom of the news release
           Caution: This is not always available
        Args:
            soup_article (BeautifulSoup): Soup of the given link
            r_link (str): single link of the news release
            r_date (str): single release date
            r_type (str): single release type

        Returns:
            dict: list and a bigger dict with descriptive information
        """
        # also get the Meta data at the very end of the webpage
        # watch out: is not always available
        metadata_webpage_bottom = []
        if soup_article.find('table', {"class": "news_footer_layout"}):
            for line in soup_article.find('table', {"class": "news_footer_layout"}).find_all("tr"):
                metadata_webpage_bottom.append(line.text.strip().replace("\n", " "))
        # Save the metadata in a dict
        metadata_webpage_bottom_dict = {}
        for item in metadata_webpage_bottom:
            if ":" in item:
                key, value = item.split(': ')
                metadata_webpage_bottom_dict[key] = value
        return metadata_webpage_bottom, metadata_webpage_bottom_dict


    def get_release_df(self, r_link: str, r_date: str, r_type: str, title_dict: dict,
                 release_body_title: list, disc_text: list, release_text: list,
                 release_text_after_contact: str, metadata_webpage_bottom_dict: dict,
                 header_meta_data, metadata_webpage_bottom, save_pdf: int):
        time = dt.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        # Save evrything in a pd.DataFrame for easy access
        trans_date = pd.to_datetime(r_date.split(" ")[0] + " " + str(self.month_dict[r_date.split(" ")[1]]) + " " + r_date.split(" ")[2], dayfirst=True)
        release_df = pd.DataFrame(index=range(1))
        release_df["r_link"] = r_link
        release_df["r_date"] = trans_date
        release_df["r_type"] = r_type
        release_df["r_scraped"] = time
        release_df["r_comp_name"] = title_dict["Company_Name"]
        release_df["r_title"] = title_dict["Release_Title"]
        release_df["r_body_title"] = release_body_title if release_body_title != [] else ""
        release_df["r_body_disc"] = disc_text if disc_text != [] else ""
        release_df["r_text"] = "".join(release_text) if release_text != [] else ""
        release_df["r_text_after_contact"] = "".join(release_text_after_contact)

        for it in metadata_webpage_bottom_dict.items():
            release_df[it[0]] = it[1]
        return release_df
        # # Finally save the whole scraped content in a .csv file as well
        # if save_csv:
        #     file_name = isin + "_" + trans_date.strftime("%d-%m-%Y") + ".csv"
        #     print(f"Saving file: {file_name} in path: {path_save}")
        #     release_df.to_csv(path_save + "//" + file_name)
        #     # if specified also save a .pdf file of the article
        # if save_pdf:
        #     file_name = isin + "_" + trans_date.strftime("%d-%m-%Y") + ".pdf"
        #     print(f"Saving file: {file_name} in path: {self.path_save}")
        #     try:
        #         converter.convert(r_link, self.path_save + "//" + file_name)
        #     except:
        #         print(f"Problems solving pdf file for release: {r_link.split('/')[-2]}")




def scrape_isin_list(isin_list: list, save_pdf: bool, save_csv: bool,
                     path_save: str, num_to_scrape: int, month_dict: dict):
    """Wrapper function for using the EQS scraper class for a multitude of isins
       for each isin the the given amount of articles is scraped from the EQS
       News Website

    Args:
        isin_list (list): List of isins for which the articles should be scraped
        save_pdf (bool): If True saves the website article also as pdf document
        save_csv (bool): If True saves a dataframe like .csv file for easy access
        path_save (str): Respective path into into which the files should be saved
        num_to_scrape (int): Number of articles for each isin
        month_dict (dict): Part of the settings, Matches each month to its number
    """
    for isin_ in isin_list:
        path_save_new = path_save + "//" + isin_
        scraper = EQS_Scraper(isin_, save_pdf, save_csv, path_save_new,
                              num_to_scrape, month_dict)
        links, dates, types = scraper.scrape_inital_link()
        soup_list = scraper.getSoupFromLink(links)

        news_df = None
        for s, r, d, t in zip(soup_list, links, dates, types):
            r_comp_name, r_comp_name_err = scraper.getTitle_CompanyName(s, r, d, t)
            r_body_title = scraper.getReleaseBodyTitle(s, r)
            r_disc_txt = scraper.getReleaseDiscText(s, r)
            r_body, r_body_err, r_body_ac, r_body_ac_err = scraper.getReleaseBodyText(s, r, d, t)
            r_bottom, r_bottom_dict = scraper.getMetaDataBottom(s, r, d, t)
            r_header, r_header_err = scraper.getHeader_Meta_Data(s, r, d, t)
            # Finally save all the scraped data as .csv and/or pdf
            release_df = scraper.get_release_df(r, d, t, r_comp_name, r_body_title, r_disc_txt, r_body,
                            r_body_ac, r_bottom_dict, r_header, r_bottom, save_pdf)
            if news_df is None:
                news_df = release_df
            else:
                news_df = pd.concat([news_df, release_df], axis=0)
                
        news_df.to_parquet(path_save + "/" + f"{isin_}.parquet")


if __name__ == '__main__':
    save_csv = True
    path_save = "D:/esq_scraped"
    num_to_scrape = 500

    isin_list = ["DE0007164600", "DE0007236101", "DE000BAY0017", "DE0008404005", "DE000BASF111",
                "DE0005557508", "DE0007100000", "DE0007664005"]
    scrape_isin_list(isin_list=isin_list, 
                     save_pdf=True, 
                     save_csv=save_csv,
                     path_save=path_save, 
                     num_to_scrape=num_to_scrape,
                     month_dict=month_dict)