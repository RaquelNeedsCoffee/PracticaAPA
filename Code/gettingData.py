import requests


def get_data():
    sorted_pro_matches = []
    parameters = {}
    num_reqs = 5
    for i in range(0, num_reqs):
        if len(sorted_pro_matches) > 0:
            parameters = {'less_than_match_id': sorted_pro_matches[0]['match_id']}
        open_dota = requests.get('https://api.opendota.com/api/proMatches', params=parameters)
        sbm = sorted(open_dota.json(), key=lambda k: k['match_id'])
        sorted_pro_matches = sbm + sorted_pro_matches
        print(sorted_pro_matches[0]['match_id'], sorted_pro_matches[-1]['match_id'], len(sorted_pro_matches))


get_data()
