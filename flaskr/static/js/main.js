$(document).ready(function () {
    $(".dropdown-trigger").dropdown();
    $('.sidenav').sidenav();
});

function onLoad() {
    console.log("onLOad()");
    gapi.load('auth2', function () {
        gapi.auth2.init();
    });
}

//Google signOut
function signOut() {
    var auth2 = gapi.auth2.getAuthInstance();
    auth2.signOut().then(function () {
        console.log("Sign out");
        window.location.replace('/auth/logout');
    });
}