$.get("data/table.json", function(data){
    $("#searchbutton").click(function(d){
        radarChange(data);
    });
})