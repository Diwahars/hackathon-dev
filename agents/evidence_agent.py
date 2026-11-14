from tools.tavily_client import search_web

def run(query, time_range=None):
    return search_web(query, time_range=time_range)
