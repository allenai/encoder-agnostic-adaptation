CANONICAL_FORMS = {
    "no_relation": ["NR"],
    "org:founded_by": ["{obj}, founder of {subj}.",
                       "{obj}, who established {subj}.",
                       "{subj}, founded by {obj}.",
                       "{obj} was the founder of {subj}.",
                       "{subj} founder {obj}."],
    "per:employee_of": ["{subj} is an employee of {obj}.",
                        "{subj}, member of {obj}.",
                        "{subj} is a member of {obj}.",
                        "{subj} joined {obj}.",
                        "{subj}, spokeman of {obj}."],
    "org:alternate_names": ["{subj} known as {obj}.",
                            "{subj}, formally known as {obj}.",
                            "{subj}, then called {obj}.",
                            "Called {subj} or {obj}."],
    "per:cities_of_residence": ["{subj} lived in {obj}.",
                                "{subj} moved to {obj}.",
                                "{subj}'s home in {obj}.",
                                "{subj} grew up in {obj}.",
                                "{subj} who lived in {obj}."],
    "per:children": ["{subj}'s child is {obj}.",
                     "{subj}'s daughter is {obj}.",
                     "{subj} has given birth to a son, {obj}."],
    "per:title": ["{subj} is a {obj}."],
    "per:siblings": ["{subj}'s sibling is {obj}.",
                     "{subj}'s sister, {obj}.",
                     "{obj}, {subj}'s brother."],
    "per:religion": ["{subj}, a {obj}.",
                     "{subj}, a {obj} minister.",
                     "{subj} minister {obj}."],
    "per:age": ["{subj} is {obj} years old.",
                "{subj} dies at age {obj}.",
                "{subj}, aged {obj}.",
                "{subj} reached the age of {obj}."],
    "org:website": ["Find {subj} online in {obj}."],
    "per:stateorprovinces_of_residence": ["{subj} lived in {obj}.",
                                          "{subj} moved to {obj}.",
                                          "{subj}'s home in {obj}.",
                                          "{subj} grew up in {obj}.",
                                          "{subj} who lived in {obj}."],
    "org:member_of": ["{subj} is part of {obj}.",
                      "{subj} has join the {obj}.",
                      "{subj} is a member of {obj}.",
                      "{obj} is composed of {subj}."],
    "org:top_members/employees": ["{obj} is the head of {subj}.",
                                  "{subj} CEO {obj}.",
                                  "{obj}, the CEO of {subj}.",
                                  "{obj}, {subj}'s president.",
                                  "{obj}, president of {subj}.",
                                  "{obj} heads {subj}."],
    "per:countries_of_residence": ["{subj} lived in {obj}.",
                                   "{subj} moved to {obj}.",
                                   "{subj}'s home in {obj}.",
                                   "{subj} grew up in {obj}.",
                                   "{subj} who lived in {obj}."],
    "org:city_of_headquarters": ["{subj}, based in {obj}.",
                                 "{subj} is headquartered in {obj}.",
                                 "{subj}, an organization based in {obj}.",
                                 "{subj}, which is based in {obj}."],
    "org:members": ["{obj} is part of {subj}.",
                    "{obj} has join the {subj}.",
                    "{obj} is a member of {subj}.",
                    "{subj} is composed of {obj}."],
    "org:country_of_headquarters": ["{subj}, based in {obj}.",
                                    "{subj}, based in Dublin, {obj}.",
                                    "{subj} is headquartered in {obj}.",
                                    "{subj}, an organization based in {obj}.",
                                    "{subj}, which is based in {obj}."],
    "per:spouse": ["{subj} is married to {obj}.",
                   "{subj} married {obj}.",
                   "{subj}'s wife {obj}.",
                   "{subj} and her husband {obj}."],
    "org:stateorprovince_of_headquarters": ["{subj}, based in {obj}.",
                                            "{subj}, based in Seattle, {obj}.",
                                            "{subj} is headquartered in {obj}.",
                                            "{subj}, an organization based in {obj}.",
                                            "{subj}, which is based in {obj}."],
    "org:number_of_employees/members": ["{subj} employes {obj} workers.",
                                        "{subj} is an organization with {obj} employees.",
                                        "{subj} has {obj} employees."],
    "org:parents": ["{subj}, a unit of {obj}.",
                    "{subj} at {obj}.",
                    "{subj} is a division of {obj}.",
                    "{subj} was sold to {obj}."],
    "org:subsidiaries": ["{obj}, a unit of {subj}.",
                         "{obj} at {subj}.",
                         "{obj} is a division of {subj}.",
                         "{obj} was sold to {subj}."],
    "per:origin": ["{subj} is a {obj} native.",
                   "{obj} {subj}.",
                   "{subj} is a {obj}."],
    "org:political/religious_affiliation": ["{obj} group {subj}."],
    "per:other_family": ["{subj} and {obj} are family members.",
                         "{subj}'s uncle {obj}.",
                         "{subj}'s aunt {obj}.",
                         "{subj}'s grandmother {obj}.",
                         "{subj}'s grandfather {obj}.",
                         "{subj}'s niece {obj}."],
    "per:stateorprovince_of_birth": ["{subj} was born in {obj}.",
                                     "{subj} was born on January 1st in {obj}.",
                                     "{subj} was born in London, {obj}."],
    "org:dissolved": ["{subj} was dissolved in {obj}.", 
                      "{subj} announced bankrupcy in {obj}."],
    "per:date_of_death": ["{subj} died in {obj}.",
                          "{subj} died at his home in {obj}."],
    "org:shareholders": ["{obj} acquired some of {subj}.",
                         "{obj} invested in {subj}.",
                         "{subj}'s shareholder {obj}."],
    "per:alternate_names": ["{subj}, who was known as {obj}.",
                            "{subj}, whose real name is {obj}.",
                            "{subj}, then known as {obj}."],
    "per:parents": ["{obj} is {subj}'s parent.",
                    "{subj}'s father, {obj}.",
                    "{obj}, {subj}'s father.",
                    "{obj}, mother of {subj}."],
    "per:schools_attended": ["{subj} graduated from {obj}.",
                             "{subj} received a degree from {obj}.",
                             "{subj} attended {obj}."],
    "per:cause_of_death": ["{subj} died of {obj}.",
                           "{subj} died from {obj}."],
    "per:city_of_death": ["{subj} died in {obj}.",
                          "{subj} died at his home in {obj}.",
                          "{subj} died at Sunday in {obj}."],
    "per:stateorprovince_of_death": ["{subj} died in {obj}.",
                                     "{subj} died at his home in {obj}.",
                                     "{subj} died in London, {obj}.",
                                     "{subj} died at Sunday in {obj}."],
    "org:founded": ["{subj} was established in {obj}.",
                    "Founded {subj} in {obj}.",
                    "{subj}, established in {obj}.",
                    "The founder founded {subj} in {obj}."],
    "per:country_of_birth": ["{subj} was born in {obj}.",
                             "{subj} was born on January 1st in {obj}.",
                             "{subj} was born in Berlin, {obj}."],
    "per:date_of_birth": ["{subj} was born in {obj}.",
                          "{subj} was born on {obj}."],
    "per:city_of_birth": ["{subj} was born in {obj}.",
                          "{subj} was born on January 1st in {obj}."],
    "per:charges": ["{subj} was convicted of {obj}.",
                    "{subj} face {obj} among other charges."],
    "per:country_of_death": ["{subj} died in {obj}.",
                             "{subj} died at his home in {obj}.",
                             "{subj} died in London, {obj}.",
                             "{subj} died at Sunday in {obj}."]
}
