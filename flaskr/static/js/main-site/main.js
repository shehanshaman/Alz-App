//csv read
$('#chooseFile').bind('change', function () {
  var filename = $("#chooseFile").val();
  if (/^\s*$/.test(filename)) {
    $(".file-upload").removeClass('active');
    $("#noFile").text("No file chosen..."); 
  }
  else {
    $(".file-upload").addClass('active');
    $("#noFile").text(filename.replace("C:\\fakepath\\", "")); 
  }
});

$(function (){
  $("#listTo").sortable();
  $("#listTo").disableSelection();
  
  $(document).on("click", "#listFrom li", function () {
    $(this).unbind("click").appendTo("#listTo");
  });
  $(document).on("click", "#listTo li", function () {
    $(this).unbind("click").appendTo("#listFrom");
  });
});

var slider = document.getElementById("myRange");
var output = document.getElementById("demo");
output.innerHTML = slider.value;

slider.oninput = function() {
  output.innerHTML = this.value;
}