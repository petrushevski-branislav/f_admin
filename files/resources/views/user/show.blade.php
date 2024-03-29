@extends('layouts.app')

@section('content')
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">{{ $user->name }}</div>

                    <div class="card-body">
                        <form action="/users/documents" method="post" enctype="multipart/form-data">
                            @csrf
                            <div class="row">
                                <div class="col-6">
                                    Choose file:
                                </div>
                                <div class="col-6">
                                    <input type="file" accept="*" name="textFile"/>
                                </div>
                            </div>
                            <div class="row mt-2">
                                <div class="col-6">
                                    File name:
                                </div>
                                <div class="col-6">
                                    <input type="name" name="name"/>
                                </div>
                            </div>
                            <input type="hidden" name="userId" value="{{$user->id}}"/>
                            <div class="row mt-2 text-left">
                                <div class="col-6 offset-6">
                                    <button type="submit" class="btn btn-sm btn-primary">Upload file</button>
                                </div>
                            </div>
                        </form>
                    </div>
                </div>

                @foreach($documents as $document)
                    <div class="card mt-4">
                        <div class="card-header">Document name: {{$document->name}}</div>

                        <div class="card-body">
                            <a href="{{$document->data}}">Download file</a>
                        </div>

                        <div class="card-footer">
                            <form action="/users/{{$user->id}}/documents/{{$document->_id}}" method="post">
                                @method('DELETE')
                                @csrf
                                <button type="submit" class="btn btn-sm btn-primary">Delete file</button>
                            </form>
                        </div>

                    </div>
                @endforeach
            </div>
        </div>
    </div>
@endsection
