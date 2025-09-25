import requests

class GoogleSearch:
    def __init__(self, api_key, base_url):
        self.base_url = base_url
        self.headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "google-search72.p.rapidapi.com"
    }
        
    def search(self, query: str = ""):
        query = query.strip()

        if query == "":
            raise ValueError("请输入问题")
        
        querystring = {"q":query,"lr":"zh-cn","num":"3"}

        response = requests.get(self.base_url, headers=self.headers, params=querystring)
        
        print()
        
        data_list = response.json()['items']
        
        if len(response.json()['items']) == 0:
            return ""
        
        else:
            result_arr = []
            result_str = ""
            for i in range(len(response.json()['items'])):
                item = data_list[i]
                title = item["title"]
                description = item["snippet"]
                item_str = f"{title}: {description}"
                print(item_str)
                result_arr.append(item_str)
            result_str = "\n".join(result_arr)
        return result_str

if __name__ == "__main__":
    from dotenv import load_dotenv
    import os 
    load_dotenv()
    api_key = os.getenv("GOOGLE_SEARCH_API")
    base_url = "https://google-search72.p.rapidapi.com/search"
    querystring = {"q":"人工智能神经网络","lr":"zh-cn","num":"10"}
    api_key = "1633ca2dc0msh1febf4794ed5bd1p1bbb28jsn02c7fd8068c5"
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "google-search72.p.rapidapi.com"
    }
    
    search = GoogleSearch(base_url, headers)
    res = search.search('周杰伦')
    print(res)