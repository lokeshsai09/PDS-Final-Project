// Constantly Update Table
setInterval(function() {
    $.getJSON('/songs', function(data) {
        CreateHtmlTable(data);
        console.log(data,"DATA");
      });
    return false;
}, 5000);

function CreateHtmlTable(data) {
  //Clear result div
  $("#ResultArea").html("");
  //Crate table html tag
  var table = $("<table id='myTable' class= 'display stripe row-border hover'></table>").appendTo("#ResultArea");
 
  //Create table header row
  var thead = $("<thead></thead>").appendTo(table);

  var rowHeader = $("<tr></tr>").appendTo(thead);
  
  $("<td></td>").text("Name").appendTo(rowHeader);
  
  $("<td></td").text("Album").appendTo(rowHeader);
  
    $("<td></td>").text("Artist").appendTo(rowHeader)
  var tbody = $("<tbody></tbody>").appendTo(table);
  //Get JSON data by calling action method in controller
  $.each(data, function (i, value) {

      //Create new row for each record
      
      var row = $("<tr></tr>").appendTo(tbody);
      $("<td></td>").text(value.name).appendTo(row);
      $("<td></td>").text(value.album).appendTo(row);
      $("<td></td>").text(value.artist).appendTo(row);
  });
  $('#myTable').DataTable({
    "pageLength": 25,
    "className": "customColor"
  } );
}