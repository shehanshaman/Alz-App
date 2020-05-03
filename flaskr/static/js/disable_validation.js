
div_list = ['visualization_icon', 'preprocess_icon', 'feature_icon', 'analize_icon', 'validation_icon', 'modeling_icon', 'prediction_icon'];

function disable_read_fun(disable_prop){
	// console.log(disable_prop);
	for(var i = 0; i<=disable_prop.length ; i++){
	    if( disable_prop[i] == 0 ){
	        $('#' + div_list[i]).css("pointer-events", "none");
	        $('#' + div_list[i]).css("opacity", "0.4"); 

	        $('#nav_' + div_list[i]).css("pointer-events", "none");
	        $('#nav_' + div_list[i]).css("opacity", "0.4");

	        $('#side_nav_' + div_list[i]).css("pointer-events", "none");
	        $('#side_nav_' + div_list[i]).css("opacity", "0.4");     

			$('#side_step_nav_' + div_list[i]).css("pointer-events", "none");
	        $('#side_step_nav_' + div_list[i]).css("opacity", "0.4");             
	    }
	}	
}
