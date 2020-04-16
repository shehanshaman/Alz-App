$(document).ready(function () {
    $(".dropdown-trigger").dropdown();
    $('.sidenav').sidenav();
});

//Google signOut
function signOut() {
    window.location.replace('/auth/logout');
}

function goBack() {
    window.history.back();
}