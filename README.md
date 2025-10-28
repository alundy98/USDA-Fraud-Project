not getting Q and A stuff:
https://ask.fdic.gov/fdicinformationandsupportcenter/s/article/Q-What-is-the-Grandparent-Scam?language=en_US

This Python project scrapes articles from the FDIC website related to fraud, scams, and deceptive practices. It automatically extracts the article title, URL, full publication date, year, and the article text from both HTML pages and PDFs. Articles older than 10 years are automatically ignored. The results are saved to a CSV file (final.csv) for easy analysis or further processing. The scraper uses requests, BeautifulSoup, pdfplumber, and regular expressions to handle varied date formats and document types.


#To run on your own:
If you want to alter the query, just go to that same link*(the search page) and input your keywords or filter(-inactive, etc..) and then put that
link as the base_url, if it breaks double check that the xpath and css are exactly correct to the path in the html.
Ignore the js
None of the Information and support center links work, the search bar and containers mess with the scraper. if Needed, we can write a helper 
explicitly for q and a pages, though I think thats not necessary.
Also, several links were already dead before scraping, so those are removed

No more Selenium-> the js loading wasn't actually in the way of what we needed to get, so ignoring it entirely
and just directly putting the xpath or css for the html items we needed worked better, but slower.
No API: I got an api key, but I guess it was access to their bank records and stuff like that, couldn't get it to accept my key
and scrape the articles. Might be possible and I just didn't do it right, but I tried for ages and got nowhere.

Supabase Schema:
Pk -> url: text
title:text
year:float8, double precision float
full_date: text
text: text
scraped_at: timestampz