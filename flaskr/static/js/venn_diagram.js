
function hover_white(p_div, svg_div){
  $(svg_div + "," + p_div).hover(function(){
        $(p_div).css("background-color", "transparent");
        $(p_div).css("color", "white");
        $(svg_div).css("fill", "#26a69a");
    }, function(){
        $(p_div).css("background-color", "transparent");    
        $(p_div).css("color", "#26a69a");
        $(svg_div).css("fill", "white");
  });   
}

function hover_white_blue(p_div, svg_div){
  $(svg_div + "," + p_div).hover(function(){
        $(p_div).css("background-color", "transparent");
        $(p_div).css("color", "white");
        $(svg_div).css("fill", "#26a69a");
    }, function(){
        $(p_div).css("background-color", "transparent");    
        $(p_div).css("color", "#26a69a");
        $(svg_div).css("fill", "#9df6ed");
  });   
}

hover_white('.p_M', '#M_clc');
hover_white('.p_R', '#R_clc');
hover_white('.p_L', '#L_clc');
hover_white('.p_T', '#T_clc');
hover_white_blue('.p_LRM', '#LRM_clc');
hover_white_blue('.p_TRM', '#TRM_clc');
hover_white_blue('.p_TLM', '#TLM_clc');

$('.modal-content-1 .close').click(function(){
    $('#popup_gene_list').slideToggle("hide");
    $('.M_gene').hide(0);
    $('.T_gene').hide(0);
    $('.R_gene').hide(0);
    $('.L_gene').hide(0);
    $('.TR_gene').hide(0);
    $('.TL_gene').hide(0);
    $('.LR_gene').hide(0);
});

$('.p_M, #M_clc').click(function(){
    $('#popup_gene_list').slideToggle("show");
    $('.M_gene').show(0);
    $('#tourControls').hide(0);
});

$('.p_R, #R_clc').click(function(){
    $('#popup_gene_list').slideToggle("show");
    $('.R_gene').show(0);
    $('#tourControls').hide(0);
});


$('.p_L, #L_clc').click(function(){
    $('#popup_gene_list').slideToggle("show");
    $('.L_gene').show(0);
    $('#tourControls').hide(0);
});

$('.p_T, #T_clc').click(function(){
    $('#popup_gene_list').slideToggle("show");
    $('.T_gene').show(0);
    $('#tourControls').hide(0);
});

$('.p_LRM, #LRM_clc').click(function(){
    $('#popup_gene_list').slideToggle("show");
    $('.LR_gene').show(0);
    $('#tourControls').hide(0);
});


$('.p_TRM, #TRM_clc').click(function(){
    $('#popup_gene_list').slideToggle("show");
    $('.TR_gene').show(0);
    $('#tourControls').hide(0);
});

$('.p_TLM, #TLM_clc').click(function(){
    $('#popup_gene_list').slideToggle("show");
    $('.TL_gene').show(0);
    $('#tourControls').hide(0);
});

$('body').click(function(evt){    
    if($(evt.target).is('#popup_gene_list')) {
        $('#popup_gene_list').slideToggle("hide");
        $('.M_gene').hide(0);
        $('.T_gene').hide(0);
        $('.R_gene').hide(0);
        $('.L_gene').hide(0);
        $('.TR_gene').hide(0);
        $('.TL_gene').hide(0);
        $('.LR_gene').hide(0);
    }
});