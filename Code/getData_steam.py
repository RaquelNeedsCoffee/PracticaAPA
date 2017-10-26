import requests

api_key = 'D2621F001696D18D41FC7C955EF66E40'
steam_api = 'http://api.steampowered.com/IDOTA2Match_570'


# solo deja 500 (?)

def get_data():
    sorted_pro_matches = []
    parameters = {'key': api_key}
    num_reqs = 1
    for i in range(0, num_reqs):
        if len(sorted_pro_matches) > 0:
            parameters['start_at_match_id'] = sorted_pro_matches[0]['match_id'] - 1
        new_matches = sorted(
            requests.get(steam_api + '/GetMatchHistory/v1', params=parameters)
            .json()['result']['matches'],
            key=lambda k: k['match_id']
        )
        print(i, len(new_matches))
        sorted_pro_matches = new_matches + sorted_pro_matches

    return sorted_pro_matches


matches = get_data()

match_ids = []
for m in matches:
    match_ids = match_ids + [m['match_id']]

print(matches[0]['match_id'], len(matches), len(set(match_ids)))
print(matches[0])

parameters = {'match_id': matches[0]['match_id']}
match_det = requests.get(steam_api + '/GetMatchDetails/v1', params=parameters)
print(match_det)
