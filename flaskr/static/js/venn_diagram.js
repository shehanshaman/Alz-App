
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
    $('#popup_gene_list').hide(500);
    $('.M_gene').hide(0);
    $('.T_gene').hide(0);
    $('.R_gene').hide(0);
    $('.L_gene').hide(0);
    $('.TR_gene').hide(0);
    $('.TL_gene').hide(0);
    $('.LR_gene').hide(0);
});

$('.p_M, #M_clc').click(function(){
    $('#popup_gene_list').fadeIn(500);
    $('.M_gene').show(0);
});

$('.p_R, #R_clc').click(function(){
    $('#popup_gene_list').fadeIn(500);
    $('.R_gene').show(0);
});


$('.p_L, #L_clc').click(function(){
    $('#popup_gene_list').fadeIn(500);
    $('.L_gene').show(0);
});

$('.p_T, #T_clc').click(function(){
    $('#popup_gene_list').fadeIn(500);
    $('.T_gene').show(0);
});

$('.p_LRM, #LRM_clc').click(function(){
    $('#popup_gene_list').fadeIn(500);
    $('.LR_gene').show(0);
});


$('.p_TRM, #TRM_clc').click(function(){
    $('#popup_gene_list').fadeIn(500);
    $('.TR_gene').show(0);
});

$('.p_TLM, #TLM_clc').click(function(){
    $('#popup_gene_list').fadeIn(500);
    $('.TL_gene').show(0);
});
