@extends('layouts.app')

@section('content')
    <div class="container">
        <table class="table table-striped">
            <thead>
            <tr>
                <th>Name</th>
                <th>Email</th>
                <th>File</th>
            </tr>
            </thead>
            <tbody>
            @foreach($users as $user)
                <tr>
                    <td>{{$user->name}}</td>
                    <td>{{$user->email}}</td>
                    <td>
                        <form action="/users/documents" method="post" enctype="multipart/form-data">
                            @csrf
                            <input type="file" name="textFile"/>
                            <input type="hidden" name="userId" value="{{$user->id}}}"/>
                            <button type="submit" class="btn btn-sm btn-primary">Upload file</button>
                        </form>
                    </td>
                </tr>
            @endforeach
            </tbody>
        </table>
    </div>
    </div>
@endsection
