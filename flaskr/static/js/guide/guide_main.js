
    $(document).ready(function () {
        $('.tour').on('click', function () {
            $.aSimpleTour(tour);
        });
    });

    data = $('#user_want_tour_p').text();
    notification_on = (data == 1);
    
    if (notification_on) {
        $('.notification_pop_on').show(0);
        $('.notification_pop_off').hide(0);

        $(document).ready(function(){
            $(".tour_create").click();
            $("#tourNext").click();
            $("#tourPrev").click();
            $("#tourEnd").click();
            $(".tour_create").click(); 
        });

    }
    else {
        $('.notification_pop_on').hide(0);
        $('.notification_pop_off').show(0);

        $(document).ready(function(){
            $(".tour_create").click();
            $("#tourNext").click();
            $("#tourPrev").click();
            $("#tourEnd").click();
        });
    }

    user_id = $('#user_id_p').text();
    $('.notification_pop').click(function () {

        $.get("/update/user/tour/", {id: user_id, tour: !notification_on}, function (data, status) {
            notification_on = (data == 'true');
            
            if (notification_on) {
                $('.notification_pop_on').show(0);
                $('.notification_pop_off').hide(0);

                $(document).ready(function(){
                    $(".tour_create").click(); 
                }); 
            }
            else {
                $('.notification_pop_on').hide(0);
                $('.notification_pop_off').show(0);
                $('#tourControls').hide(0);
            }
        });

    });

function guide_main_call(tip_data, controls_position){

    var tour = {
        data: tip_data,
        welcomeMessage: 'Lets Start Tour!',
        controlsPosition: controls_position,
        buttons: {
            next: {text: 'Next &rarr;', class: 'btn btn-default'},
            prev: {text: '&larr; Previous', class: 'btn btn-default'},
            start: {text: 'Start', class: 'btn btn-primary'},
            end: {text: 'End', class: 'btn btn-primary'}
        },
        controlsCss: {
            background: 'rgba(255, 255, 255, 0.71)',
            color: '#1c1d21',
            width: '400px',
            border: '2px solid #26a69a'
        }
    };

    return tour;    
}