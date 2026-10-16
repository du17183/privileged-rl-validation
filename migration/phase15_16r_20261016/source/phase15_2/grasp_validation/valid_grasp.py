"""GT measurements and frozen-train support; never consume future outcomes."""
def normal_entry(planner,i,raw):
    return planner.profile_data.entry(raw,planner,i)
