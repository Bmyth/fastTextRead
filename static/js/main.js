$(function() {

    $('.btn').click(function(){
        var text = $('.text').val();
        $.ajax({
            url: '/api/taxonomy',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({text:text}),
            success: function(res){
                result = res.result;
                $('.result').text(result.prediction);
            }
        });
    })
})