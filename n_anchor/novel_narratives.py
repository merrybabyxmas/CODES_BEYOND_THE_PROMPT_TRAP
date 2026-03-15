"""
Novel Narratives from Project Gutenberg (PG-19) for N-Anchor Experiments
=========================================================================
Pre-computed anchors and translated prompts for 5 real novel scenes.

Sources: Alice in Wonderland, Frankenstein, Pride and Prejudice,
         The Adventures of Sherlock Holmes

All anchor descriptions and translated prompts are deterministically
pre-defined (NO API calls, NO proxy) for full reproducibility.
"""

NOVEL_NARRATIVES = [
    # ── Scene 1: Alice's Adventures in Wonderland ──
    {
        'id': 'novel_alice_wonderland',
        'title': "Alice's Adventures in Wonderland (Lewis Carroll)",
        'source': 'PG-19 / Project Gutenberg #11',
        'anchor': {
            'entity': (
                'A young girl about 7 years old with long straight blonde hair '
                'held back by a black headband, wearing a light blue knee-length '
                'dress with a white pinafore apron, white stockings, and black '
                'Mary Jane shoes'
            ),
            'background': (
                'A fantastical Victorian cottage interior with oversized furniture, '
                'a tiny wooden door in the wall, checkered floor tiles, whimsical '
                'colorful decorations, soft warm light filtering through small windows'
            ),
        },
        'raw_sentences': [
            '"If I eat one of these cakes," she thought, "it\'s sure to make '
            'some change in my size; and as it can\'t possibly make me larger, '
            'it must make me smaller, I suppose."',
            'So she swallowed one of the cakes, and was delighted to find that '
            'she began shrinking directly.',
            'As soon as she was small enough to get through the door, she ran '
            'out of the house, and found quite a crowd of little animals and '
            'birds waiting outside.',
            'The poor little Lizard, Bill, was in the middle, being held up by '
            'two guinea-pigs, who were giving it something out of a bottle.',
            'They all made a rush at Alice the moment she appeared; but she ran '
            'off as hard as she could, and soon found herself safe in a thick wood.',
        ],
        'translated_prompts': [
            ('A young girl with long blonde hair and black headband, wearing a '
             'light blue dress with white pinafore, looks thoughtfully at small '
             'colorful frosted cakes on a wooden table in a fantastical Victorian '
             'room with oversized furniture and a tiny door, warm light, '
             'cinematic, high quality'),
            ('A young girl with long blonde hair and black headband, wearing a '
             'light blue dress with white pinafore, eats a small cake and begins '
             'magically shrinking in size in a fantastical Victorian room, '
             'sparkles around her, whimsical atmosphere, cinematic, high quality'),
            ('A tiny young girl with long blonde hair in a light blue dress runs '
             'through a small wooden door and emerges into a garden where a crowd '
             'of small dressed animals and colorful birds wait outside, fantastical '
             'setting, lush green, cinematic, high quality'),
            ('A small green lizard in a waistcoat being held up by two fluffy '
             'guinea-pigs who give it medicine from a glass bottle, surrounded '
             'by small dressed animals in a whimsical garden with oversized '
             'flowers, cinematic, high quality'),
            ('A young girl with long blonde hair in a light blue dress with '
             'white pinafore runs through a whimsical garden away from small '
             'animals, entering a thick dark ancient wood with tall mossy trees '
             'and dappled golden light, cinematic, high quality'),
        ],
    },

    # ── Scene 2: Frankenstein - Arctic Expedition ──
    {
        'id': 'novel_frankenstein_arctic',
        'title': 'Frankenstein (Mary Shelley) - Arctic Scene',
        'source': 'PG-19 / Project Gutenberg #84',
        'anchor': {
            'entity': (
                'A middle-aged sea captain with a weathered tanned face and '
                'salt-and-pepper beard, wearing a dark navy wool greatcoat with '
                'brass buttons, a captain\'s tricorn hat, and brown leather gloves'
            ),
            'background': (
                'A wooden three-masted sailing ship trapped in vast Arctic ice '
                'fields, white and blue ice stretching to the horizon, grey '
                'overcast sky, cold blue-white atmosphere, icy dark water between '
                'ice floes'
            ),
        },
        'raw_sentences': [
            'About two hours after this occurrence we heard the ground sea, and '
            'before night the ice broke and freed our ship.',
            'We, however, lay to until the morning, fearing to encounter in the '
            'dark those large loose masses which float about after the breaking '
            'up of the ice.',
            'I profited of this time to rest for a few hours.',
            'In the morning, however, as soon as it was light, I went upon deck '
            'and found all the sailors busy on one side of the vessel, apparently '
            'talking to someone in the sea.',
            'It was, in fact, a sledge, like that we had seen before, which had '
            'drifted towards us in the night on a large fragment of ice.',
        ],
        'translated_prompts': [
            ('A wooden three-masted sailing ship in the Arctic as massive ice '
             'fields crack and break apart around the hull, dark icy water '
             'rushing through gaps, grey overcast sky, cold blue-white light, '
             'dramatic atmosphere, cinematic, high quality'),
            ('A wooden sailing ship resting in dark Arctic waters at night, '
             'large loose ice chunks floating around the vessel, a captain with '
             'salt-and-pepper beard in a dark navy coat watches from the deck '
             'railing, moonlight on ice, cinematic, high quality'),
            ('A middle-aged sea captain with weathered face and salt-and-pepper '
             'beard rests in a cramped wooden ship cabin below deck, dim lantern '
             'light swaying, wooden bunks and nautical instruments on the wall, '
             'Arctic cold, cinematic, high quality'),
            ('A middle-aged sea captain with salt-and-pepper beard in a dark navy '
             'coat walks onto the wooden deck in cold morning light, finding '
             'sailors crowded along the railing looking down at the icy Arctic '
             'sea below, grey sky, cinematic, high quality'),
            ('A wooden dog sledge sits abandoned on a large fragment of floating '
             'Arctic ice near a sailing ship, cold pale morning light, vast white '
             'ice fields, grey sky, sailors on deck watch the mysterious sledge, '
             'cinematic, high quality'),
        ],
    },

    # ── Scene 3: Frankenstein - Young Victor at the Inn ──
    {
        'id': 'novel_frankenstein_inn',
        'title': 'Frankenstein (Mary Shelley) - Discovery Scene',
        'source': 'PG-19 / Project Gutenberg #84',
        'anchor': {
            'entity': (
                'A young boy of about 13 with pale skin and dark wavy hair '
                'falling over his forehead, wearing a simple white linen shirt '
                'with rolled sleeves and a brown wool vest, intense curious dark eyes'
            ),
            'background': (
                'A cozy old Swiss inn during a heavy rainstorm, low wooden beams '
                'on the ceiling, a crackling stone fireplace, dusty bookshelves '
                'along rough stone walls, warm candlelight, rain pattering on '
                'small leaded windows'
            ),
        },
        'raw_sentences': [
            'When I was thirteen years of age we all went on a party of pleasure '
            'to the baths near Thonon; the inclemency of the weather obliged us '
            'to remain a day confined to the inn.',
            'In this house I chanced to find a volume of the works of Cornelius '
            'Agrippa.',
            'I opened it with apathy; the theory which he attempts to demonstrate '
            'and the wonderful facts which he relates soon changed this feeling '
            'into enthusiasm.',
            'A new light seemed to dawn upon my mind, and bounding with joy, I '
            'communicated my discovery to my father.',
            'My father looked carelessly at the title page of my book and said, '
            '"Ah! Cornelius Agrippa!"',
        ],
        'translated_prompts': [
            ('A young boy of 13 with dark wavy hair in a white linen shirt and '
             'brown vest sits bored in a cozy old Swiss inn during a rainstorm, '
             'low wooden beams, crackling stone fireplace, rain on leaded windows, '
             'warm candlelight, cinematic, high quality'),
            ('A young boy with dark wavy hair in a white linen shirt discovers '
             'an old leather-bound book on a dusty shelf in a cozy inn, candlelight '
             'illuminating the worn cover, rough stone walls and low wooden beams, '
             'cinematic, high quality'),
            ('A young boy with dark wavy hair reads an old book with growing '
             'excitement, his eyes widening with wonder, sitting by a crackling '
             'stone fireplace in a cozy inn, warm candlelight on his fascinated '
             'face, cinematic, high quality'),
            ('A young boy with dark wavy hair jumps up from his chair with joy, '
             'clutching an old book, running eagerly to his father across the '
             'cozy inn room, warm firelight, wooden beams, enthusiastic expression, '
             'cinematic, high quality'),
            ('An older distinguished gentleman in a dark wool coat looks '
             'dismissively at the title page of an old book held by his excited '
             'young son with dark wavy hair, by the fireplace in a cozy inn, '
             'candlelight, cinematic, high quality'),
        ],
    },

    # ── Scene 4: Pride and Prejudice - Bingley's Visit ──
    {
        'id': 'novel_pride_prejudice',
        'title': 'Pride and Prejudice (Jane Austen)',
        'source': 'PG-19 / Project Gutenberg #1342',
        'anchor': {
            'entity': (
                'A genteel English lady in her early 40s with light brown hair '
                'pinned up under a white lace bonnet, wearing a long Regency-era '
                'dress in pale green muslin with a high empire waistline, '
                'pearl earrings and a cameo brooch'
            ),
            'background': (
                'A grand English country house interior, an elegant Regency-era '
                'drawing room with tall sash windows overlooking manicured gardens, '
                'striped silk wallpaper, mahogany furniture, a writing desk, warm '
                'afternoon sunlight streaming in'
            ),
        },
        'raw_sentences': [
            '"If I can but see one of my daughters happily settled at Netherfield," '
            'said Mrs. Bennet to her husband, "and all the others equally well '
            'married, I shall have nothing to wish for."',
            'In a few days Mr. Bingley returned Mr. Bennet\'s visit, and sat about '
            'ten minutes with him in his library.',
            'He had entertained hopes of being admitted to a sight of the young '
            'ladies, of whose beauty he had heard much; but he saw only the father.',
            'The ladies were somewhat more fortunate, for they had the advantage '
            'of ascertaining, from an upper window, that he wore a blue coat and '
            'rode a black horse.',
            'An invitation to dinner was soon afterwards despatched; and already '
            'had Mrs. Bennet planned the courses that were to do credit to her '
            'housekeeping, when an answer arrived which deferred it all.',
        ],
        'translated_prompts': [
            ('A genteel English lady in her 40s with brown hair under a white '
             'lace bonnet, wearing a pale green Regency muslin dress, speaks '
             'animatedly with hand gestures to her husband seated in an armchair '
             'in an elegant drawing room with tall windows, afternoon sunlight, '
             'cinematic, high quality'),
            ('A young well-dressed gentleman in a blue tailcoat and white cravat '
             'sits politely across from an older gentleman in a wood-paneled '
             'library of a grand English country house, tall mahogany bookshelves, '
             'warm light, Regency era, cinematic, high quality'),
            ('A young gentleman in a blue tailcoat walks through the hallway of '
             'a grand English country house, glancing around curiously, Regency-era '
             'interior with striped wallpaper and portraits, tall windows, warm '
             'afternoon light, cinematic, high quality'),
            ('Several young ladies in white and pastel Regency-era muslin dresses '
             'peer excitedly from an upper window of a grand English country house, '
             'looking down at a gentleman riding a black horse on the gravel drive, '
             'afternoon sunlight, cinematic, high quality'),
            ('A genteel English lady in a pale green Regency dress reads a letter '
             'with a disappointed expression in an elegant drawing room, formal '
             'dining table being set by servants in the background, warm afternoon '
             'light through tall windows, cinematic, high quality'),
        ],
    },

    # ── Scene 5: Sherlock Holmes - Baker Street ──
    {
        'id': 'novel_sherlock_holmes',
        'title': 'The Adventures of Sherlock Holmes (Arthur Conan Doyle)',
        'source': 'PG-19 / Project Gutenberg #1661',
        'anchor': {
            'entity': (
                'A Victorian gentleman doctor in his early 30s with a neat dark '
                'mustache, wearing a dark brown Harris tweed three-piece suit '
                'with a gold watch chain, a bowler hat, and carrying a wooden '
                'walking cane'
            ),
            'background': (
                '221B Baker Street apartment in Victorian London, dark patterned '
                'wallpaper, a cluttered desk covered in papers and scientific '
                'instruments, a warm fireplace with two worn leather armchairs, '
                'gaslight lamps casting amber glow, foggy London rooftops visible '
                'through the window'
            ),
        },
        'raw_sentences': [
            '"And good-night, Watson," he added, as the wheels of the royal '
            'brougham rolled down the street.',
            '"If you will be good enough to call tomorrow afternoon at three '
            'o\'clock I should like to chat this little matter over with you."',
            'At three o\'clock precisely I was at Baker Street, but Holmes had '
            'not yet returned.',
            'The landlady informed me that he had left the house shortly after '
            'eight o\'clock in the morning.',
            'I sat down beside the fire, however, with the intention of awaiting '
            'him, however long he might be.',
        ],
        'translated_prompts': [
            ('A Victorian gentleman in a dark brown tweed suit and bowler hat '
             'stands on a dimly lit cobblestone London street at night, waving '
             'goodbye as a royal horse-drawn carriage rolls away, gas street '
             'lamps glowing in thick fog, cinematic, high quality'),
            ('A tall thin man in a deerstalker cap and dark coat speaks to a '
             'gentleman in brown tweed on a foggy Victorian London doorstep, '
             'gas lamp illuminating their faces, 221B Baker Street door number '
             'visible, cinematic, high quality'),
            ('A Victorian gentleman in a dark brown tweed suit enters the 221B '
             'Baker Street apartment, finding it empty, cluttered desk with '
             'papers and scientific instruments, dark wallpaper, gaslight lamps, '
             'afternoon light through window, cinematic, high quality'),
            ('An elderly landlady in a Victorian dark dress with white apron '
             'stands at the doorway of 221B Baker Street apartment, speaking '
             'to a gentleman visitor in brown tweed, narrow hallway with dark '
             'wallpaper, gaslight, cinematic, high quality'),
            ('A Victorian gentleman in a dark brown tweed suit sits in a worn '
             'leather armchair beside a crackling fireplace at 221B Baker Street, '
             'waiting patiently, gaslight lamps, cluttered bookshelves, foggy '
             'London rooftops through the window, cinematic, high quality'),
        ],
    },
]
