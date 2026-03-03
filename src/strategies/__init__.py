"""
strategies/
-----------
One module per client selection strategy.

Available strategies:
    random   ->  strategies/random_selector.py
    oort     ->  strategies/oort_selector.py
    divfl    ->  strategies/divfl_selector.py
    fairfl   ->  strategies/fairfl_selector.py

Each strategy implements:
    select(round_num, client_stats) -> List[int]   (list of selected client IDs)
    update(round_num, results)      -> None         (feed back training results)

See individual files for full API documentation.
"""
