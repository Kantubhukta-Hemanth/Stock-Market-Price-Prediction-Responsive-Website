function darkmode() {
    var element = document.body;
    element.classList.toggle("dark-mode");
}

function goback() {
    window.history.back();
}

function urlexist(link) {
    var http = new XMLHttpRequest();
    http.open('HEAD', link, false)
    http.send()
    return http.status != 404
}

function submit() {
    var grp = document.getElementById("group").value
    var sem = document.getElementById("semi").value
    var link = "class_sub/" + grp.toLowerCase() + sem + ".html"
    if (urlexist(link)) {
        document.location = link
    }
    else {
        document.location = '/Home/error.html'
    }

}