import dota2api

api = dota2api.Initialise("D2621F001696D18D41FC7C955EF66E40")


# SOLO DEJA COJER 500 MATCHES!!!!!!!!!!!!

def get_data():
    sorted_pro_matches = []
    num_reqs = 7
    for i in range(0, num_reqs):
        if len(sorted_pro_matches) > 0:
            start_id = sorted_pro_matches[0]['match_id'] - 1
            new_matches = sorted(
                api.get_match_history(start_at_match_id=start_id)['matches'],
                key=lambda k: k['match_id']
            )
        else:
            new_matches = sorted(
                api.get_match_history()['matches'],
                key=lambda k: k['match_id']
            )
        print(new_matches[0]['match_id'], new_matches[-1]['match_id'], len(new_matches))
        sorted_pro_matches = new_matches + sorted_pro_matches

    return sorted_pro_matches


matches = get_data()

match_ids = []
for m in matches:
    match_ids = match_ids + [m['match_id']]

print(matches[0]['match_id'], '\n', len(matches), len(set(match_ids)))

print(matches[0]['match_id'], matches[-1]['match_id'], len(matches))
print(matches[0])
