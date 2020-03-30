$(document).ready(function () {
    $(".dropdown-trigger").dropdown();

    gapi.load('auth2', function () {
        gapi.auth2.init();
    });
});
