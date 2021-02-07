@extends('layouts.app')

@section('content')
    <div class="container">
        <table class="table table-striped">
            <thead>
            <tr>
                <th>Name</th>
                <th>Email</th>
            </tr>
            </thead>
            <tbody>
            @foreach($users as $user)
                <tr>
                    <td>{{$user->name}}</td>
                    <td>{{$user->email}}</td>>
                    <td>
                        <a href="users/{{$user->id}}">Details</a>
                    </td>
                </tr>
            @endforeach
            </tbody>
        </table>
    </div>
    </div>
@endsection
