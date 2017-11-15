import requests
import dota2api

api_key = 'D2621F001696D18D41FC7C955EF66E40'
opendota_api_URL = 'https://api.opendota.com/api'
dota2api_api = dota2api.Initialise(api_key)


def get_pro_matches_ids():
    sorted_pro_matches = sorted(
                requests.get(opendota_api_URL + '/proMatches').json(),
                key=lambda k: k['match_id'])
    if len(sorted_pro_matches) > 0:
        parameters = {'less_than_match_id': sorted_pro_matches[0]['match_id']}
        num_requests = 10
        for i in range(0, num_requests):
            sorted_new_matches = sorted(
                requests.get(opendota_api_URL + '/proMatches', params=parameters).json(),
                key=lambda k: k['match_id']
            )
            sorted_pro_matches = sorted_new_matches + sorted_pro_matches

    return sorted_pro_matches


def get_match_data(match_id):
    match_data = {}
    match_data_opendota = requests.get(opendota_api_URL + '/matches/' + str(match_id))
    if match_data_opendota.status_code == 200:
        match_data['opendota'] = match_data_opendota.json()
    try:
        match_data['dota2api'] = dota2api_api.get_match_details(match_id=match_id)
    except Exception:
        pass

    return match_data


pro_matches_ids = get_pro_matches_ids()
pro_match_data = get_match_data(pro_matches_ids[0]['match_id'])
print(pro_matches_ids[0].keys())
print(pro_match_data.keys())
for k, v in pro_match_data.items():
    print(k, v.keys())
