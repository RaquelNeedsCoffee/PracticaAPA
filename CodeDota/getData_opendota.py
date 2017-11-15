import requests

opendota_api_URL = 'https://api.opendota.com/api'


def get_data():
    sorted_pro_matches = []
    parameters = {}
    num_reqs = 5
    for i in range(0, num_reqs):
        if len(sorted_pro_matches) > 0:
            parameters = {'less_than_match_id': sorted_pro_matches[0]['match_id']}
            pro_matches = requests.get(opendota_api_URL + '/proMatches', params=parameters)
        sorted_new_matches = sorted(pro_matches.json(), key=lambda k: k['match_id'])
        sorted_pro_matches = sorted_new_matches + sorted_pro_matches
        # print(len(sorted_new_matches), len(sorted_pro_matches))

    return sorted_pro_matches


matches = get_data()

match_ids = []
for m in matches:
    match_ids = match_ids + [m['match_id']]

print(matches[0]['match_id'], matches[-1]['match_id'], len(matches), len(set(match_ids)))
match_keys = matches[0].keys()

print()
for k in match_keys:
    print(k)
