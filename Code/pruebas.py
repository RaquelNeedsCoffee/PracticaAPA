import requests


def get_data():
    sorted_pro_matches = []
    parameters = {'key': "D2621F001696D18D41FC7C955EF66E40"}
    num_reqs = 5
    for i in range(0, num_reqs):
        if len(sorted_pro_matches) > 0:
            parameters['start_at_match_id'] = sorted_pro_matches[0]['match_id'] - 1
        new_matches = sorted(
            requests.get('http://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/v1', params=parameters)
            .json()['result']['matches'],
            key=lambda k: k['match_id']
        )
        sorted_pro_matches = new_matches + sorted_pro_matches

    return sorted_pro_matches


matches = get_data()

match_ids = []
for m in matches:
    match_ids = match_ids + [m['match_id']]

print(matches[0]['match_id'], '\n', len(matches), len(set(match_ids)))
print(matches[0])
