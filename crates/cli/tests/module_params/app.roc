app [main] {
    pf: platform "../fixtures/multi-dep-str/platform/main.roc",
}

import Api { appId: "one", protocol: https } as App1
import Api { appId: "two", protocol: http } as App2

https = \url -> "https://$(url)"
http = \url -> "http://$(url)"

usersApp1 =
    # pass top-level fn in a module with params
    List.map [1, 2, 3] App1.getUser

main =
    app3Id = "three"

    import Api { appId: app3Id, protocol: https } as App3

    getUserApp3Nested = \userId ->
        # use captured params def
        App3.getUser userId

    """
    App1.baseUrl: $(App1.baseUrl)
    App2.baseUrl: $(App2.baseUrl)
    App3.baseUrl: $(App3.baseUrl)
    App1.getUser 1: $(App1.getUser 1)
    App2.getUser 2: $(App2.getUser 2)
    App3.getUser 3: $(App3.getUser 3)
    App1.getPost 1: $(App1.getPost 1)
    App2.getPost 2: $(App2.getPost 2)
    App3.getPost 3: $(App3.getPost 3)
    App1.getPosts [1, 2]: $(Inspect.toStr (App1.getPosts [1, 2]))
    App2.getPosts [3, 4]: $(Inspect.toStr (App2.getPosts [3, 4]))
    App2.getPosts [5, 6]: $(Inspect.toStr (App2.getPosts [5, 6]))
    App1.getPostComments 1: $(App1.getPostComments 1)
    App2.getPostComments 2: $(App2.getPostComments 2)
    App2.getPostComments 3: $(App2.getPostComments 3)
    App1.getCompanies [1, 2]: $(Inspect.toStr (App1.getCompanies [1, 2]))
    App2.getCompanies [3, 4]: $(Inspect.toStr (App2.getCompanies [3, 4]))
    App2.getCompanies [5, 6]: $(Inspect.toStr (App2.getCompanies [5, 6]))
    usersApp1: $(Inspect.toStr usersApp1)
    getUserApp3Nested 3: $(getUserApp3Nested 3)
    """
