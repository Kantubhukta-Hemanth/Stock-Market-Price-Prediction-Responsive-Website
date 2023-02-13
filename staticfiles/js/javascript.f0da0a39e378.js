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


function showSuggestions() {
    var keyword = JSON.parse('{{ keywords | safe }}');
    //var tickers = JSON.parse("{{ tickers | safe }}");
    var input = document.getElementById("searchInput").value;
    var suggestions = [];
    
    for (var i = 0; i < keyword.length; i++) {
        key = keyword[i]
        if (key.toLowerCase().includes(input.toLowerCase())) {

            suggestions.push(key);
            if (suggestions.length >= 10) {
                break;
            }
        }
    }
    
    
    var suggestionList = document.getElementById("suggestions");
    suggestionList.innerHTML = "";
    for (var i = 0; i < suggestions.length; i++) {
        var suggestion = document.createElement("div");
        suggestion.innerHTML = suggestions[i];
        suggestion.onclick = function () {
            document.getElementById("searchInput").value = this.innerHTML;
            suggestionList.innerHTML = "";
            $('#suggestions').hide();
        };
        suggestionList.appendChild(suggestion);
    }
}

function verify_stock() {
    var keyword = JSON.parse('{{ keywords | safe }}');
    var input = document.getElementById("searchInput").value;
    console.log(input)
    if (!keyword.includes(input)) {
        alert('Stock not Found...!');
    }
}
