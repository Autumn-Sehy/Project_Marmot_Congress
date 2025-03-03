import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_processing import load_data_parallel
from utils import get_female_mcs, get_male_mcs, get_mcs_regions, get_dems, get_repubs

mcs_regions = {k.lower(): v for k, v in get_mcs_regions().items()}
male_mcs = [mc.lower() for mc in get_male_mcs()]
female_mcs = [mc.lower() for mc in get_female_mcs()]
dem_list = [d.lower() for d in get_dems()]
rep_list = [r.lower() for r in get_repubs()]


def dem_and_repub_metadata(party):
    party = party.lower()
    if party in ['democrat', 'democratic']:
        return 'Democrat'
    elif party in ['republican', 'freedom caucus']:
        return 'Republican'
    return 'Other'


def export_for_tfjs(vectors, metadata, output_dir='tfjs_export', prefix='all'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vectors_path = os.path.join(output_dir, f'{prefix}_vectors.tsv')
    with open(vectors_path, 'w', encoding='utf-8') as f:
        for vector in vectors:
            f.write('\t'.join(map(str, vector)) + '\n')

    metadata_path = os.path.join(output_dir, f'{prefix}_metadata.tsv')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        f.write('MC\tParty\tGender\tRegion\n')
        for meta in metadata:
            mc_name = meta["mc"]
            party = dem_and_repub_metadata(meta["party"])
            gender = meta["gender"]
            region = meta["region"]
            f.write(f'{mc_name}\t{party}\t{gender}\t{region}\n')


def create_mc_vectors(include_republicans=True, include_democrats=True, include_freedom_caucus=False):
    w_in = np.load("results_and_embeddings/w_in_torch.npy")
    with open("results_and_embeddings/word_to_index.pkl", "rb") as f:
        word_to_index = pickle.load(f)

    word2vec = {word: w_in[idx] for word, idx in word_to_index.items()}

    data_dir = "Data"
    texts, labels, mcs = load_data_parallel(data_dir)
    print(f"Loaded {len(texts)} documents.")
    mc_texts = {}
    party_label = {}

    missing_regions = set()

    for text, label, mc in zip(texts, labels, mcs):
        mc_lower = mc.lower()
        if mc_lower not in mc_texts:
            mc_texts[mc_lower] = []
            party_label[mc_lower] = label
        mc_texts[mc_lower].append(text)

    mc_vecs = []
    metadata = []

    for mc, texts_list in mc_texts.items():
        if party_label[mc].lower() == "unknown":
            if mc in dem_list:
                party_label[mc] = "Democrat"
            elif mc in rep_list:
                party_label[mc] = "Republican"

        normalized_party = dem_and_repub_metadata(party_label[mc])
        if (normalized_party == "Republican" and not include_republicans) or \
                (normalized_party == "Democrat" and not include_democrats) or \
                (normalized_party == "Other" and not include_freedom_caucus):
            continue

        combined_text = " ".join(texts_list)
        words = combined_text.split()
        vecs = []

        if mc in male_mcs:
            gender = "M"
        elif mc in female_mcs:
            gender = "F"
        else:
            gender = "U"
            print(f"{mc} was not matched as male or female.")

        region = mcs_regions.get(mc, "Unknown")
        if region == "Unknown":
            missing_regions.add(mc)

        for w in words:
            w = w.lower()
            if w in word2vec:
                vecs.append(word2vec[w])

        if vecs:
            avg_vec = np.mean(vecs, axis=0)
            mc_vecs.append(avg_vec)
            metadata.append({
                "mc": mc,
                "party": party_label[mc],
                "gender": gender,
                "region": region
            })

    if missing_regions:
        print(f"WARNING: {len(missing_regions)} members of Congress have unknown regions:")
        for mc in sorted(missing_regions):
            print(f"  - '{mc}'")
        print("Add these members to the get_mcs_regions() dictionary in your utils.py file.")

    scaler = StandardScaler()
    normalized_mc_vectors = scaler.fit_transform(np.array(mc_vecs))
    return normalized_mc_vectors, metadata


def build_vectors(include_republicans=True, include_democrats=False, include_freedom_caucus=False):
    suffix = ""
    if include_republicans and include_democrats:
        suffix = "all"
    elif include_republicans:
        suffix = "republicans"
    elif include_democrats:
        suffix = "democrats"
    elif include_freedom_caucus:
        suffix = "freedom_caucus"
    else:
        suffix = "custom"

    normalized_mc_vectors, metadata = create_mc_vectors(
        include_republicans=include_republicans,
        include_democrats=include_democrats,
        include_freedom_caucus=include_freedom_caucus
    )

    output_dir = 'tensorflow_information'
    export_for_tfjs(
        vectors=normalized_mc_vectors,
        metadata=metadata,
        output_dir=output_dir,
        prefix=suffix
    )

    print(f"Generated embeddings for {len(metadata)} speakers in '{output_dir}'")
    return normalized_mc_vectors, metadata


def main():
    build_vectors(include_republicans=True, include_democrats=False)


if __name__ == '__main__':
    main()