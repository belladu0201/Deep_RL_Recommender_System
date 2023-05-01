def create_feature_matrix(
    sorted_events,
    n_files=2,
    path_name="./item_properties_part",
    one_hot_encode=True,
    top_features=500,
):
    """Helper function to create a one hot encoded feature matrix from the Retail Rocket item properties dataset."""
    for i in range(n_files):
        if i == 0:
            item_features = pd.read_csv(path_name + str(i + 1) + ".csv")
        else:
            item_features = pd.concat(
                [item_features, pd.read_csv(path_name + str(i + 1) + ".csv")],
                ignore_index=True,
            )
    item_features = item_features[
        item_features["itemid"].isin(sorted_events["item_id"].unique().tolist())
    ].drop_duplicates()
    item_features["property_value"] = (
        item_features["property"].str.strip() + item_features["value"].str.strip()
    )
    item_features = item_features.drop(["timestamp"], axis=1).drop_duplicates()

    if one_hot_encode:
        # one hot encode the item features
        one_hot_encoded = pd.DataFrame()
        itemids = []
        event_item_list = sorted_events.item_id.unique()
        event_item_list.sort()
        item_list = item_features["itemid"].unique()
        properties = (
            item_features["property_value"]
            .value_counts()
            .head(top_features)
            .index.tolist()
        )
        for item in event_item_list:
            if len(itemids) % 1000 == 0:
                print(round(len(itemids) / len(event_item_list) * 100, 2), "% done")
            if item not in item_list:
                one_hot_encoded = pd.concat(
                    [one_hot_encoded, pd.DataFrame(np.zeros(len(properties))).T],
                    ignore_index=True,
                )
                itemids.append(item)
                continue
            item_properties = item_features[item_features["itemid"] == item][
                "property_value"
            ].unique()
            one_hot_encoded = pd.concat(
                [
                    one_hot_encoded,
                    pd.DataFrame(
                        [1 if x in item_properties else 0 for x in properties]
                    ).T,
                ],
                ignore_index=True,
            )
            itemids.append(item)

        print("100% done")

        return one_hot_encoded, itemids
    else:
        return item_features, item_features["itemid"].unique().tolist()
