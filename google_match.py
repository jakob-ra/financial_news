import json
import urllib

def google_KG_match(query, api_key, type=None):
    """ Returns dictionary with info of best match to query from Google knowledge graph API. Type can be
    e.g. People, Corporation, Website. """
    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {'query': query, 'limit': 1, 'indent': True, 'key': api_key, 'types': type}
    url = service_url + '?' + urllib.parse.urlencode(params)
    try:
        response = json.loads(urllib.request.urlopen(url).read())
    except HTTPError as e:
        print(e)
    best_response = next(iter(response['itemListElement']), None)
    # if best_response != None:
    #     res = best_response['result']['name']
    # else:
    #     res = None
    # print(best_response)
    # res = {k: best_response['result'][k] for k in ['name', 'url', 'description']}
    # res['score'] = int(best_response['resultScore'])

    return best_response

# api_key = open('google_api_key.txt', 'r').read()
# print(google_KG_match('biogen idec', api_key, type='Corporation'))
# print(google_KG_match('columbia university medical center', api_key, type='Corporation'))