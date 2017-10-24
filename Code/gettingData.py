import dota2api
import json
import requests
api = dota2api.Initialise("D2621F001696D18D41FC7C955EF66E40")
hist = api.get_match_history(account_id=41231571)
match = api.get_match_details(match_id=1000193456)
leages = api.get_league_listing()
matches = api.get_match_history(matches_requested=1000)
print(len(matches['matches']))

print('\n')
print(match.keys())
print(match['game_mode_name'])
print(match['radiant_name'])
#dict_keys(['players', 'dire_name', 'dire_captain', 'radiant_captain', 'barracks_status_dire', 'positive_votes', 'radiant_team_id', 'game_mode', 'start_time', 'dire_team_complete', 'lobby_name', 'pre_game_duration', 'picks_bans', 'game_mode_name', 'lobby_type', 'match_seq_num', 'duration', 'negative_votes', 'barracks_status_radiant', 'dire_score', 'radiant_win', 'radiant_logo', 'flags', 'engine', 'match_id', 'first_blood_time', 'radiant_score', 'tower_status_dire', 'dire_team_id', 'cluster_name', 'human_players', 'radiant_name', 'tower_status_radiant', 'cluster', 'dire_logo', 'leagueid', 'radiant_team_complete'])

parameters = {'key': "D2621F001696D18D41FC7C955EF66E40", 'matches_requested': 30}

newmatches = requests.get('http://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/v1',params=parameters)
#IDOTA2Match_570/GetMatchHistory/v001/
print(len(newmatches.json()['result']['matches']))
#print(newmatches.content.decode("utf-8"))

opendota = requests.get('https://api.opendota.com/api/proMatches')
sbm = sorted(opendota.json(), key=lambda k: k['match_id'])
print(sbm[0]['match_id'])
# for i in sbm:
#     print(i['match_id'])
#     print('\n')

for i in range(0, 100):
    parameters = {'less_than_match_id': sbm[0]['match_id']}
    #TODO: mirar si funciona con la api oficial
    newmatches = requests.get('http://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/v1',params=parameters)
    opendota = requests.get('https://api.opendota.com/api/proMatches',params=parameters)
    l = len(sbm)
    sbm = sorted(opendota.json(), key=lambda k: k['match_id'])
    print(sbm[0]['match_id'], sbm[-1]['match_id'], l)

