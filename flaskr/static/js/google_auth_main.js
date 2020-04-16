$(document).ready(function () {
    $(".dropdown-trigger").dropdown();
    $('.sidenav').sidenav();
});

function onLoad() {
    gapi.load('auth2', function () {
        gapi.auth2.init();
    });
}

//Google signOut
function signOut() {
    var auth2 = gapi.auth2.getAuthInstance();
    auth2.signOut().then(function () {
        window.location.replace('/auth/logout');
    });
}

function goBack() {
    window.history.back();
}