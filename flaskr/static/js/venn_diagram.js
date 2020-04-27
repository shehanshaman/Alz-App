gene_list = []

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
/*add here*/

function set_gene_list(gene_list_post){

    gene_list_string = gene_list_post.substring(2, gene_list_post.length-2);
    gene_list = gene_list_string.split('\', \'');

    gene_have_list = $('.gene_ex_list_div').text();
    gene_have_string = gene_have_list.substring(2, gene_have_list.length-2);
    gene_have_new_list = gene_have_string.split('\', \'');

    $(".gene_list_div").empty();
    $(".gene_list_div").append('<li class="active"><a href="#1b" data-toggle="tab">' + gene_list[0] + '</a></li>');
    
    for( var i=1; i<gene_list.length ; i++){
        $(".gene_list_div").append('<li><a id="' + i + '" href="#1b" data-toggle="tab">' + gene_list[i] + '</a></li>');
    }

    index_name = gene_list[0];

    if(is_gene_exsist(gene_have_new_list, index_name)){

        $('.gene_not_have').hide(0);
        $('.gene_have').show(0);

        $('#td_name').html(gene_describe_json[index_name]["Full_name_from_nomenclature_authority"]);
        $('#td_type').html(gene_describe_json[index_name]["Feature_type"]);
        $('#td_id').html(gene_describe_json[index_name]["GeneID"]);
        $('#td_locus_tag').html(gene_describe_json[index_name]["LocusTag"]);
        $('#td_date').html(gene_describe_json[index_name]["Modification_date"]);
        $('#td_nomenclature_status').html(gene_describe_json[index_name]["Nomenclature_status"]);
        $('#td_description').html(gene_describe_json[index_name]["description"]);
        $('#td_synonyms').html(gene_describe_json[index_name]["Synonyms"]);
        $('#td_chromosome').html(gene_describe_json[index_name]["chromosome"]);
        $('#td_dbXrefs').html(gene_describe_json[index_name]["dbXrefs"]);
        $('#td_designations').html(gene_describe_json[index_name]["Other_designations"]);
        $('#td_gene_type').html(gene_describe_json[index_name]["type_of_gene"]); 
    }
    else{
        $('.gene_not_have').show(0);
        $('.gene_have').hide(0);
    }

    $(".gene_list_div li a").click(function () {

        index_name = $(this).text();

        if(is_gene_exsist(gene_have_new_list, index_name)){
            
            $('.gene_not_have').hide(0);
            $('.gene_have').show(0);
            // console.log(gene_describe_json[index_name])
            $('#td_name').html(gene_describe_json[index_name]["Full_name_from_nomenclature_authority"]);
            $('#td_type').html(gene_describe_json[index_name]["Feature_type"]);
            $('#td_id').html(gene_describe_json[index_name]["GeneID"]);
            $('#td_locus_tag').html(gene_describe_json[index_name]["LocusTag"]);
            $('#td_date').html(gene_describe_json[index_name]["Modification_date"]);
            $('#td_nomenclature_status').html(gene_describe_json[index_name]["Nomenclature_status"]);
            $('#td_description').html(gene_describe_json[index_name]["description"]);
            $('#td_synonyms').html(gene_describe_json[index_name]["Synonyms"]);
            $('#td_chromosome').html(gene_describe_json[index_name]["chromosome"]);
            $('#td_dbXrefs').html(gene_describe_json[index_name]["dbXrefs"]);
            $('#td_designations').html(gene_describe_json[index_name]["Other_designations"]);
            $('#td_gene_type').html(gene_describe_json[index_name]["type_of_gene"]);             
        }
        else{
            $('.gene_not_have').show(0);
            $('.gene_have').hide(0);
        }
      
    });

}

function is_gene_exsist(gene_have_new_list, gene_name){

    for(var j = 0; j<gene_have_new_list.length; j++){
        if( gene_have_new_list[j] == gene_name ){
            return true;
        }
    }
    return false;
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

    gene_list = $('.M_gene_list').text();
    set_gene_list(gene_list);
});

$('.p_R, #R_clc').click(function(){
    $('#popup_gene_list').slideToggle("show");
    $('.R_gene').show(0);
    $('#tourControls').hide(0);

    gene_list = $('.R_gene_list').text();
    set_gene_list(gene_list);
});


$('.p_L, #L_clc').click(function(){
    $('#popup_gene_list').slideToggle("show");
    $('.L_gene').show(0);
    $('#tourControls').hide(0);

    gene_list = $('.L_gene_list').text();
    set_gene_list(gene_list);
});

$('.p_T, #T_clc').click(function(){
    $('#popup_gene_list').slideToggle("show");
    $('.T_gene').show(0);
    $('#tourControls').hide(0);

    gene_list = $('.T_gene_list').text();
    set_gene_list(gene_list);
});

$('.p_LRM, #LRM_clc').click(function(){
    $('#popup_gene_list').slideToggle("show");
    $('.LR_gene').show(0);
    $('#tourControls').hide(0);

    gene_list = $('.LR_gene_list').text();
    set_gene_list(gene_list);
});


$('.p_TRM, #TRM_clc').click(function(){
    $('#popup_gene_list').slideToggle("show");
    $('.TR_gene').show(0);
    $('#tourControls').hide(0);

    gene_list = $('.TR_gene_list').text();
    set_gene_list(gene_list);
});

$('.p_TLM, #TLM_clc').click(function(){
    $('#popup_gene_list').slideToggle("show");
    $('.TL_gene').show(0);
    $('#tourControls').hide(0);

    gene_list = $('.TL_gene_list').text();
    set_gene_list(gene_list);
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