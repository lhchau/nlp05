validation_pairs = [
    # Synonyms or semantically similar pairs (label = 1)
    ('king', 'queen', 1),
    ('man', 'woman', 1),
    ('car', 'vehicle', 1),
    ('apple', 'orange', 1),
    ('dog', 'cat', 1),
    ('happy', 'joyful', 1),
    ('run', 'jog', 1),
    ('strong', 'powerful', 1),
    ('fast', 'quick', 1),
    ('large', 'big', 1),
    ('small', 'tiny', 1),
    ('easy', 'simple', 1),
    ('learn', 'study', 1),
    ('teach', 'educate', 1),
    ('city', 'town', 1),
    
    # Antonyms or semantically dissimilar pairs (label = 0)
    ('man', 'dog', 0),
    ('car', 'apple', 0),
    ('king', 'car', 0),
    ('fast', 'slow', 0),
    ('happy', 'sad', 0),
    ('strong', 'weak', 0),
    ('large', 'small', 0),
    ('hot', 'cold', 0),
    ('young', 'old', 0),
    ('rich', 'poor', 0),
    ('up', 'down', 0),
    ('night', 'day', 0),
    ('light', 'dark', 0),
    ('city', 'village', 0),
    
    # Unrelated pairs (label = 0)
    ('apple', 'car', 0),
    ('dog', 'house', 0),
    ('tree', 'phone', 0),
    ('river', 'mountain', 0),
    ('computer', 'chair', 0),
    ('book', 'camera', 0),
    ('keyboard', 'bottle', 0),
    ('train', 'lamp', 0),
    ('pencil', 'table', 0),
    ('cloud', 'keyboard', 0),
    
    # More similar pairs for diversity (label = 1)
    ('teacher', 'professor', 1),
    ('car', 'automobile', 1),
    ('bicycle', 'bike', 1),
    ('child', 'kid', 1),
    ('intelligent', 'smart', 1),
    ('angry', 'furious', 1),
    ('quick', 'rapid', 1),
    ('strong', 'muscular', 1),
    ('young', 'youthful', 1),
    ('talk', 'speak', 1),
    ('see', 'watch', 1),
]